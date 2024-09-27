from dataclasses import dataclass
import logging
from typing import Dict, Optional, List, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from transformers.file_utils import ModelOutput
from rag.index_v2 import DistributedFAISSIndex
from rag.index_io import load_passages
from rag.passage_encoder import custom_jsonl_transformer
from rag.prompt_choices import PromptType
from rag.tasks.qa import Task as QATask
from gritlm import GritLM

logger = logging.getLogger(__name__)


@dataclass
class GritLMTrainOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    loss_emb: Optional[Tensor] = None
    loss_gen: Optional[Tensor] = None


class DistributedContrastiveLoss:
    def __init__(self, temperature: float, negatives_cross_device: bool):
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.temperature = temperature
        self.negatives_cross_device = negatives_cross_device        
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError('Cannot do negatives_cross_device without distributed training')
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def __call__(self, q_reps, p_reps):
        if self.negatives_cross_device:
            # This gathers both negatives and positives.
            # It could likely be optimized by only gathering negatives.
            q_reps = self._dist_gather_tensor(q_reps)
            p_reps = self._dist_gather_tensor(p_reps)
        scores = self.compute_similarity(q_reps, p_reps) / self.temperature
        scores = scores.view(q_reps.size(0), -1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target *= (p_reps.size(0) // q_reps.size(0))
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None: return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        # All tensors have the same shape, as pooling already applied to them
        dist.all_gather(all_tensors, t)

        all_tensors[self.rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2: return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

class NextTokenLoss:
    def __init__(self, vocab_size: int, loss_gen_type: str = "mixed", loss_gen_factor: float = 1.0):
        super().__init__()
        self.vocab_size = vocab_size
        # How to weight loss:
        # a) Each sample gets the same weight (e.g. used in BLOOMZ https://arxiv.org/abs/2211.01786)
        # -> This leads to shorter generations, as short outputs get the same weight as long ones
        # -> The loss curves for this are unstable if there's noisy or very hard short samples
        # b) Each token gets the same weight
        # -> This leads to longer generations, as long outputs get more weight than short ones
        # b.1) Each token gets the same weight globally across batches
        # -> Just sum the loss as is, optionally divide it by a constant. If using Adam, the scale
        # of the loss doesn't matter, so this is only to balance with another loss like in our case.
        # b.2) Each token gets the same weight per batch
        # -> Divide by the number of tokens in the batch
        # Problem: If distributed training, needs all gather of number of tokens on each process        
        # c) Mix of a) and b) which is what you do if you use the loss in transformers as is and 
        # then do grad acc/multi-gpu with bs>1 
        # (https://github.com/huggingface/transformers/issues/24725; https://github.com/pytorch/pytorch/issues/72047)
        self.loss_gen_factor = loss_gen_factor
        self.loss_gen_type = loss_gen_type
        if loss_gen_type == "token": # b.1)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
        elif loss_gen_type == "mixed": # c)
            self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        else:
            raise ValueError(f"Invalid loss_gen_type: {loss_gen_type}")
        
    def __call__(self, labels, logits):
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        # Normalize by number of non-ignored tokens
        if self.loss_gen_type == "token":
            return (self.cross_entropy(shift_logits, shift_labels) / labels.size(0)) * self.loss_gen_factor
        elif self.loss_gen_type == "mixed":
            return self.cross_entropy(shift_logits, shift_labels) * self.loss_gen_factor


class GritLMTrainModel(GritLM):
    TRANSFORMER_CLS = AutoModel
    def __init__(
        self,
        temperature: float = 1.0,
        negatives_cross_device: bool = False,
        loss_gen_type: str = "mixed",
        loss_gen_factor: float = None,
        **kwargs,
    ):
        super().__init__(**kwargs, is_inference=False)
        self.emb_loss_fn = DistributedContrastiveLoss(temperature, negatives_cross_device)
        self.gen_add_kwargs = {"return_dict": True}
        if "mixtral" in kwargs["model_name_or_path"].lower():
            logger.info("Using token loss with routing loss for mixtral")
            self.gen_loss_fn = None
            self.gen_add_kwargs["loss_gen_factor"] = loss_gen_factor
            self.gen_add_kwargs["output_router_logits"] = True
        else:
            self.gen_loss_fn = NextTokenLoss(
                self.model.config.vocab_size, loss_gen_type, loss_gen_factor
            )
        self.config = self.model.config # Required for accelerate DeepSpeed integration

    def encode(self, features):
        if features is None: return None
        # Clone to avoid modifying the original tensor
        attention_mask = features['attention_mask'].clone() if 'attention_mask' in features else None
        instruction_lens = features['instruction_lens'] if 'instruction_lens' in features else None
        kwargs = {'input_ids': features.get('input_ids'), 'attention_mask': attention_mask}

        if self.attn[:2] == 'cb':
            kwargs['instruction_lens'] = instruction_lens
        elif self.attn[:2] == 'bb':
            kwargs['is_causal'] = False
        
        out = (getattr(self.model, self.embedding_attr) if self.embedding_attr else self.model)(**kwargs)[0]

        if self.projection is not None:
            out = self.projection(out)

        # Mask out the instruction tokens for pooling
        if instruction_lens is not None:
            # Make a new copy of attention mask to prevent in-place problems
            attention_mask = features['attention_mask'].clone()
            # Mask out the instruction tokens for pooling
            for i, l in enumerate(instruction_lens):
                attention_mask[i, :l] = 0
                # Make sure not all zeros - If this happens it is a bug
                assert attention_mask[i].sum() > 0, f"All 0: {attention_mask[i]}, l: {l}"
        
        reps = self.pooling(out, attention_mask)

        
        # Normalize can change the dtype (https://discuss.pytorch.org/t/tensor-in-float16-is-transformed-into-float32-after-torch-norm/110891)
        if self.normalized: 
            in_dtype = reps.dtype
            a = torch.nn.functional.normalize(reps, dim=-1).contiguous().to(in_dtype)
            return a
        return reps.contiguous()

    def forward(
        self,
        query: Dict[str, torch.Tensor] = None,
        passage: Dict[str, torch.Tensor] = None,
        generative: Dict[str, torch.Tensor] = None,
        q_reps: Optional[torch.Tensor] = None,
        p_reps: Optional[torch.Tensor] = None,
        q_grad: bool = True,
        p_grad: bool = True,
    ):
        """
        Args:
            query: [b, n]
            passage: [b*s, m] where s is group size (usually 2)
            generative: [b, m]
        """
        # Do generative first, as emb contains an all-reduce (verified to be faster)
        if generative is not None:
            if self.gen_loss_fn is not None:
                # This pops the labels first, then the rest is passed into model                
                loss_gen = self.gen_loss_fn(
                    generative.pop('labels'), self.model(**generative, **self.gen_add_kwargs).logits
                )
            else:
                loss_gen = self.model(**generative, **self.gen_add_kwargs).loss
        else:
            loss_gen = None

        if (q_reps is None) and (query is not None):
            if q_grad:
                q_reps = self.encode(query)
            else:
                with torch.no_grad():
                    q_reps = self.encode(query)

        if (p_reps is None) and (passage is not None):
            if p_grad:
                p_reps = self.encode(passage)
            else:
                with torch.no_grad():
                    p_reps = self.encode(passage)
            
        loss_emb = self.emb_loss_fn(
            q_reps, p_reps
        ) if (q_reps is not None and p_reps is not None) else None        

        loss = sum([x for x in [loss_emb, loss_gen] if x is not None])

        # Also return q_reps in case of GradCache
        return GritLMTrainOutput(
            q_reps=q_reps,
            p_reps=p_reps,
            loss=loss,
            loss_emb=loss_emb,
            loss_gen=loss_gen,
        )

    def gradient_checkpointing_enable(self, *args, **kwargs):
        self.model.gradient_checkpointing_enable(*args, **kwargs)


class GritLMREPLUGModel(GritLMTrainModel):
    def __init__(
        self,
        temperature: float = 1.0,
        parent_model: Optional[nn.Module] = None,
        index: DistributedFAISSIndex = None,
        num_passages: int = 10,
        tokenizer: AutoTokenizer = None,
        loss_gen_type: str = "mixed",
        loss_gen_factor: float = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.temperature = temperature
        self.parent_model = parent_model
        self.num_passages = num_passages
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.tokenizer = tokenizer
        if self.parent_model is None:
            raise ValueError("parent_model must be provided for REPLUG training")

        self.lm_loss_fn = NextTokenLoss(parent_model.model.config.vocab_size, loss_gen_type, loss_gen_factor)
        self.index = index

    def retrieve_passages(self, queries: torch.Tensor) -> tuple[list[list[dict]], torch.Tensor]:
        with torch.no_grad():
            queries_np = queries.cpu().numpy()
        passages, scores = self.index.search_knn(queries_np, self.num_passages)
        return passages, torch.tensor(scores, device=queries.device)

    def forward(self, query_text: list[str], embeddings: Dict[str, torch.Tensor] = None, **kwargs):
        q_reps = self.encode(embeddings)
        retrieved_passages, retriever_scores = self.retrieve_passages(q_reps)
        
        all_lsr_scores = []
        all_retriever_scores = []

        for batch_passages, batch_scores in zip(retrieved_passages, retriever_scores):
            lsr_scores_batch = []
            
            for passage in batch_passages:
                # Prepare input: concatenate query and passage
                # print(query_text)
                # print(passage['text'])
                input_ids = self.tokenizer(query_text[0] + " " + passage['text'], return_tensors="pt", max_length=256, padding=True, truncation=True)['input_ids'].to(q_reps.device)
                
                # Compute LSR score
                # with torch.no_grad():
                outputs = self.parent_model.model(input_ids)
                lm_score = self.lm_loss_fn(input_ids, outputs.logits)
                lsr_score = torch.exp(-lm_score / self.temperature)
                lsr_scores_batch.append(lsr_score)

            # Normalize scores within the batch
            lsr_scores = F.softmax(torch.stack(lsr_scores_batch), dim=0)
            retriever_scores_norm = F.softmax(batch_scores / self.temperature, dim=0)
            
            all_lsr_scores.append(lsr_scores)
            all_retriever_scores.append(retriever_scores_norm)

        # Stack all batches
        lsr_scores = torch.stack(all_lsr_scores)
        retriever_scores = torch.stack(all_retriever_scores)

        # Compute REPLUG loss (KL-divergence)
        loss_replug = F.kl_div(retriever_scores.log(), lsr_scores, reduction='batchmean')

        # If there's a generative component, compute its loss
        # loss_gen = super().forward(generative=generative, **kwargs).loss_gen if generative is not None else None
        loss_gen = None

        # Combine losses
        loss = loss_replug + (loss_gen if loss_gen is not None else 0)

        return GritLMTrainOutput(
            q_reps=q_reps,
            p_reps=None,  # We don't have a single p_reps in this case
            loss=loss,
            loss_emb=loss_replug,
            loss_gen=loss_gen,
        )