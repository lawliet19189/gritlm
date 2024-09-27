import logging
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import HfArgumentParser, Trainer, set_seed

from gritlm.training.train_utils import get_quantization_config, get_rag_train_index

from .arguments import CustomTrainingArguments, DataArguments, ModelArguments
from .data import CustomCollator, CustomDataset, CustomRandomSampler
from .model import GritLMTrainModel, GritLMREPLUGModel
from .data_utils import get_tokenizer_and_config, get_train_dataset
from gritlm.gritlm import GritLM
from .special_tokens import BASE_BOS, TURN_SEP, USER_BOS, USER_EOS, EMBED_BOS, EMBED_EOS, ASSISTANT_BOS, ASSISTANT_EOS

# os.environ["WANDB_MODE"] = "disabled"

logger = logging.getLogger(__name__)

def args_to_dtype(args):
    if args.bf16: return torch.bfloat16
    if args.fp16: return torch.float16
    return torch.float32


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to bypass."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )

    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("Model parameters %s", model_args)
    logger.info("Data parameters %s", data_args)

    # Set seed
    set_seed(training_args.seed)

    # If embedding/unified/replug, handle grad accumulation manually inside forward of GradCacheTrainer.
    gc_chunk_size = None
    if ((training_args.gradient_accumulation_steps > 1) and \
        (training_args.negatives_cross_device) and \
        (training_args.mode in ["embedding", "unified", "replug"])) or \
        (training_args.no_gen_gas and training_args.no_emb_gas):

        gc_chunk_size = training_args.per_device_train_batch_size
        training_args.per_device_train_batch_size = \
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        training_args.gradient_accumulation_steps = 1

        logger.info("Using GradCache with chunk size %d", gc_chunk_size)
    elif (training_args.no_gen_gas or training_args.no_emb_gas):
        raise ValueError("Cannot use no_gen_gas or no_emb_gas without GradCache")


    # load tokenizer and config
    tokenizer, config = get_tokenizer_and_config(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    )
    logger.info('Config: %s', config)


    # REPLUG requires RAG setup including index, passages, and parent tokenizer
    if training_args.mode == "replug":
        assert data_args.index_path is not None, "Index path must be provided for RAG training"
        assert data_args.index_passages_path is not None, "Index passages path must be provided for RAG training"
        assert data_args.index_type is not None, "Index type must be provided for RAG training"
        assert data_args.code_size is not None, "Code size must be provided for RAG training"
        assert data_args.num_passages is not None, "Number of passages must be provided for RAG training"

        _, parent_config = get_tokenizer_and_config(
            model_args.parent_tokenizer_name if model_args.parent_tokenizer_name else model_args.parent_model_name_or_path,
            model_args.parent_config_name if model_args.parent_config_name else model_args.parent_model_name_or_path,
        )
        logger.info('Parent Config: %s', parent_config)

    # training data can either be a single file or a directory containing files
    data_files = [os.path.join(data_args.train_data, x) for x in os.listdir(data_args.train_data)] if \
        os.path.isdir(data_args.train_data) else [data_args.train_data]
    
    # Each training dataset can have a different number of samples. 
    # This is useful for example when we want to train a single model on multiple datasets with different number of samples.
    num_samples = None
    if data_args.num_samples:
        with open(data_args.num_samples, "r", encoding="utf-8") as f:
            num_samples = json.load(f)
    
    # If generative max len is not set, use the passage max len
    if data_args.generative_max_len is None:
        data_args.generative_max_len = data_args.passage_max_len

    # Iterate over all training files, load them as datasets, filter out too long instructions, and concatenate them
    ds, ds_name_to_samples, ds_embedding_lens = get_train_dataset(
        train_files=data_files,
        tokenizer=tokenizer,
        max_example_num_per_dataset=training_args.max_example_num_per_dataset,
        query_max_len=data_args.query_max_len,
        passage_max_len=data_args.passage_max_len,
        generative_max_len=data_args.generative_max_len,
        num_samples=num_samples,
        mode=training_args.mode,
    )

    # Save dataset num samples to a json file
    os.makedirs(training_args.output_dir, exist_ok=True)
    with open(os.path.join(training_args.output_dir, "dataset_num_samples.json"), "w", encoding="utf-8") as f:
        json.dump(ds_name_to_samples, f)

    if training_args.per_device_generative_bs is not None:
        assert training_args.mode == "unified", "Generative batch size is only supported in unified mode"
        assert training_args.per_device_generative_bs < training_args.per_device_train_batch_size, \
            "Generative batch size must be smaller than regular batch size"
        logger.info("Using generative batch size %d per device", training_args.per_device_generative_bs)

    
    quantization_config, load_in_4bit = get_quantization_config(qlora_enabled=training_args.qlora)

    if training_args.mode == "replug":
        index = get_rag_train_index(
            index_path=data_args.index_path,
            index_passages_path=data_args.index_passages_path,
            index_passages_num=data_args.index_passages_num,
            index_type=data_args.index_type,
            code_size=data_args.code_size,
        )

        logger.info("Loading parent model")
        frozen_model = GritLM(
            model_name_or_path=model_args.parent_model_name_or_path,
            mode="generative",
            pooling_method=model_args.pooling_method,
            normalized=model_args.normalized,
            projection=model_args.projection,
            attn=model_args.attn,
            attn_implementation=model_args.attn_implementation,
            is_inference=True,
            # negatives_cross_device=training_args.negatives_cross_device,
            temperature=training_args.parent_temperature,
            torch_dtype=torch.bfloat16, # TODO: Make this configurable
            load_in_4bit=True,
            use_cache=False,
            low_cpu_mem_usage=True,
        )
        logger.info("Loading REPLUG model")
        model = GritLMREPLUGModel(
            model_name_or_path=model_args.model_name_or_path,
            normalized=model_args.normalized,
            pooling_method=model_args.pooling_method,
            # negatives_cross_device=training_args.negatives_cross_device,
            temperature=training_args.temperature,
            mode=training_args.mode,
            projection=model_args.projection,
            attn=model_args.attn,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=args_to_dtype(training_args),
            loss_gen_type=training_args.loss_gen_type,
            loss_gen_factor=training_args.loss_gen_factor,
            use_cache=False,
            # Critical to make Mixtral work
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            load_in_4bit=load_in_4bit,
            parent_model=frozen_model,
            index=index,
            num_passages=data_args.num_passages,
            tokenizer=tokenizer,
        )
    else:
        model = GritLMTrainModel(
            model_name_or_path=model_args.model_name_or_path,
            normalized=model_args.normalized,
            pooling_method=model_args.pooling_method,
            negatives_cross_device=training_args.negatives_cross_device,
            temperature=training_args.temperature,
            mode=training_args.mode,
            projection=model_args.projection,
            attn=model_args.attn,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=args_to_dtype(training_args),
            loss_gen_type=training_args.loss_gen_type,
            loss_gen_factor=training_args.loss_gen_factor,
            use_cache=False,
            # Critical to make Mixtral work
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            load_in_4bit=load_in_4bit,
        )
    # Add special token for embed
    if model_args.pooling_method == "lasttoken":
        embed_eos = "</e>"
        if embed_eos in tokenizer.vocab:
            logger.info("Embed eos token already in vocab: %s", embed_eos)
        else:
            logger.info("Adding embed eos token to vocab: %s", embed_eos)
            tokenizer.add_tokens([embed_eos], special_tokens=True)
            model.model.resize_token_embeddings(len(tokenizer))
        config.num_vocab += 1
    else:
        embed_eos = EMBED_EOS

    if os.getenv("BIDIRECTIONAL_ATTN", False):
        if hasattr(model.model, "model"):
            model.model.model.padding_idx = tokenizer.pad_token_id
        else:
            model.model.padding_idx = tokenizer.pad_token_id

    if (training_args.lora) or (training_args.qlora):
        if training_args.qlora:
            from peft import prepare_model_for_kbit_training
            model.model = prepare_model_for_kbit_training(
                model.model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        from peft import get_peft_model, LoraConfig, TaskType
        # https://github.com/texttron/tevatron/blob/2e5d00ee21d5a7db0bd2ea1463c9150a572106d4/examples/repllama/repllama.py#L81
        # https://github.com/allenai/open-instruct/blob/9ebcb582cfc243a6dab75b4302fa432784db26c2/open_instruct/finetune.py#L478
        peft_config = LoraConfig(
            inference_mode=False, 
            r=16, 
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "o_proj", "v_proj", "k_proj", "w1", "w2", "w3"]
        )
        model.model.enable_input_require_grads()
        model.model = get_peft_model(model.model, peft_config)
        model.model.print_trainable_parameters()

    train_dataset = CustomDataset(
        ds,
        args=data_args,
        tokenizer=tokenizer,
        mode=training_args.mode,
        full_bs=training_args.per_device_train_batch_size,
        generative_bs=training_args.per_device_generative_bs,
        max_seq_len=max(data_args.query_max_len, data_args.passage_max_len, data_args.generative_max_len),
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "data_collator": CustomCollator(
            tokenizer,
            query_max_len=data_args.query_max_len,
            passage_max_len=data_args.passage_max_len,
            generative_max_len=data_args.generative_max_len,
            base_bos=BASE_BOS,
            turn_sep=TURN_SEP,
            user_bos=USER_BOS,
            user_eos=USER_EOS,
            embed_bos=EMBED_BOS,
            embed_eos=embed_eos,
            assistant_bos=ASSISTANT_BOS,
            assistant_eos=ASSISTANT_EOS,
            prefixlm=data_args.prefixlm,
            mode=training_args.mode,
            max_embed_len=model_args.emb_dim,
        ),
        "tokenizer": tokenizer,
    }

    if gc_chunk_size is not None:
        from .gradcache_trainer import GradCacheTrainer
        trainer = GradCacheTrainer(**trainer_kwargs)
        trainer.gc_chunk_size = gc_chunk_size
        trainer.emb_loss_fn = model.emb_loss_fn
        trainer.mode = training_args.mode
        trainer.no_gen_gas = training_args.no_gen_gas
        trainer.no_emb_gas = training_args.no_emb_gas
        trainer.split_emb = training_args.split_emb
        trainer.split_emb_full = training_args.split_emb_full
        trainer.emb_p_only = training_args.emb_p_only
        trainer.emb_q_only = training_args.emb_q_only
    else:
        trainer = Trainer(**trainer_kwargs)

    if len(ds_embedding_lens) > 1:
        assert training_args.dataloader_drop_last, "Multiple datasets are only supported with dropping the last incomplete batch, set `--dataloader_drop_last`"
        logger.info("Embedding dataset lengths: %s", ds_embedding_lens)
        # Multiple embedding datasets & we want to make sure each batch mostly comes from one dataset
        # Set custom sampler, see https://github.com/huggingface/transformers/blob/ccb92be23def445f2afdea94c31286f84b89eb5b/src/transformers/trainer.py#L785
        total_bs = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        total_bs = total_bs * dist.get_world_size() if dist.is_initialized() else total_bs
        trainer._get_train_sampler = lambda: CustomRandomSampler(
            total_batch_size=total_bs, ds_lens=ds_embedding_lens,
            _num_samples=sum(ds_embedding_lens), data_source=train_dataset,
        )

    if training_args.mode == "unified":
        # Track all losses
        from transformers.integrations import WandbCallback
        from transformers.integrations.integration_utils import rewrite_logs
        from transformers.trainer_pt_utils import distributed_concat
        class WandbCustomCallback(WandbCallback):
            def on_log(self, args, state, control, model=None, logs=None, **kwargs):
                if self._wandb is None: return
                if not self._initialized: self.setup(args, state, model)
                if hasattr(state, "loss_emb") and hasattr(state, "loss_gen"):
                    # Gather & avg across gpus like for actual loss
                    # https://github.com/huggingface/transformers/blob/bc72b4e2cdcbc80d5f56731f35dbc9c18b4c8de6/src/transformers/trainer.py#L2257
                    if (args.distributed_state is not None and args.distributed_state.distributed_type != "NO") or (
                        args.distributed_state is None and args.local_rank != -1):
                        state.loss_emb = distributed_concat(state.loss_emb).mean().item()
                        state.loss_gen = distributed_concat(state.loss_gen).mean().item()
                    else:
                        state.loss_emb = state.loss_emb.mean().item()
                        state.loss_gen = state.loss_gen.mean().item()
                    if state.is_world_process_zero:
                        self._wandb.log({
                            **rewrite_logs(logs),
                            "train/global_step": state.global_step,
                            "train/loss_emb": state.loss_emb,
                            "train/loss_gen": state.loss_gen,
                        })
                    del state.loss_emb
                    del state.loss_gen
                else:
                    if state.is_world_process_zero:
                        self._wandb.log({
                            **rewrite_logs(logs),
                            "train/global_step": state.global_step,
                        })

        trainer.add_callback(WandbCustomCallback())

        # Copied from below & added loss_emb/loss_gen
        # https://github.com/huggingface/transformers/blob/cc3e4781854a52cf090ffde28d884a527dab6708/src/transformers/trainer.py#L2699
        def training_step(self, model, inputs):
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                out = self.compute_loss(model, inputs, return_outputs=True)
                loss = out[0]
                loss_emb = out[1]["loss_emb"]
                loss_gen = out[1]["loss_gen"]

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                loss_emb = loss_emb.mean()
                loss_gen = loss_gen.mean()

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss) # Includes normalizing by gas

            self.state.loss_emb = getattr(self.state, "loss_emb", torch.tensor(0.0).to(loss.device))
            self.state.loss_gen = getattr(self.state, "loss_gen", torch.tensor(0.0).to(loss.device))
            self.state.loss_emb += loss_emb.detach() / self.args.gradient_accumulation_steps
            self.state.loss_gen += loss_gen.detach() / self.args.gradient_accumulation_steps
            
            return loss.detach() / self.args.gradient_accumulation_steps

        # __get__ is needed to bind the method to the Trainer instance
        trainer.training_step = training_step.__get__(trainer)

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)

    # Training
    logger.info("Starting training")
    trainer.train()
    
    # The below does not save if state dict type is `SHARDED_STATE_DICT`
    trainer.save_model()

    # To be safe do another FS save
    if (trainer.is_fsdp_enabled) and (trainer.accelerator.state.fsdp_plugin.state_dict_type != "FULL_STATE_DICT"):
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        fsd_path = os.path.join(training_args.output_dir, "full_state_dict")
        os.makedirs(fsd_path, exist_ok=True)
        trainer.save_model(fsd_path)

    # Save tokenizer & config for easy usage afterwards
    if trainer.is_world_process_zero(): 
        tokenizer.save_pretrained(training_args.output_dir)
        config.to_json_file(training_args.output_dir + "/config.json")

if __name__ == "__main__":
    main()
