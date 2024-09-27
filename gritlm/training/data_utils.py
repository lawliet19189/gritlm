import multiprocessing
import random

import datasets
from transformers import AutoTokenizer, AutoConfig
from gritlm.logger import logger
from .special_tokens import BASE_BOS, USER_BOS, USER_EOS, EMBED_BOS, ASSISTANT_BOS

def get_tokenizer_and_config(tokenizer_name, config_name):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        padding_side="right", # Has to be right so masking of instruction tokens works correctly
    )
    config = AutoConfig.from_pretrained(
        config_name,
        num_labels=1,
    )
    if not(tokenizer.pad_token) and tokenizer.bos_token:
        tokenizer.pad_token = tokenizer.bos_token
        logger.info('Set pad token to bos token: %s', tokenizer.pad_token)   

    return tokenizer, config

def filter_too_long_instructions_for_query_pos_neg(tokenizer, dataset, query_max_len, passage_max_len):
    def filter_fn(example):
        # Filter out super long examples to avoid tokenize taking forever
        query = example["query"][0]
        pos = example["pos"]
        neg = example["neg"]

        if (len(query) > query_max_len * 10) or not(example["query"][1]):
            return False
        if len(tokenizer.tokenize(BASE_BOS + USER_BOS + example["query"][0].strip("\t\n :") + USER_EOS + EMBED_BOS)) >= query_max_len:
            return False
        for ex in pos + neg:
            if (len(ex[0]) > passage_max_len * 10) or not(ex[1]):
                return False
            if len(tokenizer.tokenize(BASE_BOS + USER_BOS + ex[0].strip("\t\n :") + USER_EOS + EMBED_BOS)) >= passage_max_len:
                return False
        return True
    num_proc = max(multiprocessing.cpu_count()-2, 1) if len(dataset) > 5000 else 1
    return dataset.filter(filter_fn, num_proc=num_proc, load_from_cache_file=True)

def filter_too_long_instructions_for_text(tokenizer, dataset, generative_max_len):
    # Use passage_max_len, as this is the seq len limit for the entire generative snippet
    num_proc = max(multiprocessing.cpu_count()-2, 1) if len(dataset) > 5000 else 1
    return dataset.filter(
        lambda ex: len(tokenizer.tokenize(USER_BOS + ex["text"][0] + USER_EOS + ASSISTANT_BOS)) < generative_max_len,
        num_proc=num_proc,
        load_from_cache_file=True,
    )

def get_dataset_with_given_num_samples(dataset, num_samples, file_name):
    
    if num_samples:
        assert file_name in num_samples, f'Missing num_samples for {file_name}'
        
        dataset_len = len(dataset)
        samples = num_samples[file_name]
        if dataset_len > samples:
            return dataset.select(random.sample(list(range(dataset_len)), samples))
    return dataset

def get_train_dataset(train_files, tokenizer, max_example_num_per_dataset, query_max_len, passage_max_len, generative_max_len,num_samples, mode):
    
    ds_name_to_samples = {}
    train_ds = []
    for file in train_files:
        logger.info("Loading dataset %s", file)
        tmp_ds = datasets.load_dataset('json', data_files=file, split='train')
        tmp_ds_len = len(tmp_ds)
        # For testing, can add an origin column:
        # origin_col = [file] * len(tmp_ds)
        # tmp_ds = tmp_ds.add_column("origin", origin_col)
        if tmp_ds_len > max_example_num_per_dataset:
            tmp_ds = tmp_ds.select(
                random.sample(list(range(tmp_ds_len)), max_example_num_per_dataset)
            )
        # Check if has instructions separated such that they will be masked out later
        # If so filter out samples where the instructions are too long else they will all be 0s
        
        # For embeddings training, filter out too long instructions
        if mode in ["embedding", "unified"] and "query" in tmp_ds.features:
            
            if isinstance(tmp_ds[0]['query'], (tuple, list)):
                logger.info("Filtering out embedding samples with too long instructions for %s. Max length: %d", file, query_max_len)
                tmp_ds = filter_too_long_instructions_for_query_pos_neg(
                    tokenizer,
                    tmp_ds,
                    query_max_len,
                    passage_max_len,
                )
                tmp_ds = get_dataset_with_given_num_samples(tmp_ds, num_samples, file.split("/")[-1])

            # store num samples for each dataset & append to train datasets
            ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
            train_ds.append(tmp_ds)
        
        # For generative training, filter out too long instructions
        elif mode in ["unified", "generative", "replug"] and "text" in tmp_ds.features:

            if isinstance(tmp_ds[0]['text'], (tuple, list)):
                logger.info("Filtering out generative samples with too long instructions for %s", file)

                tmp_ds = filter_too_long_instructions_for_text(
                    tokenizer,
                    tmp_ds,
                    generative_max_len,
                )
                tmp_ds = get_dataset_with_given_num_samples(tmp_ds, num_samples, file.split("/")[-1])
            
            # store num samples for each dataset & append to train datasets
            ds_name_to_samples[file.split("/")[-1]] = len(tmp_ds)
            train_ds.append(tmp_ds)
        else:
            logger.info("Skipping dataset %s as its type could not be identified", file)
    
    embedding_dataset_lengths = []
    if mode == "embedding":
        embedding_dataset_lengths = [len(t) for t in train_ds]
        ds = datasets.concatenate_datasets(train_ds)
        logger.info("Embedding mode: %d samples", len(ds))
    elif mode in ["generative", "replug"]:
        ds = datasets.concatenate_datasets(train_ds)
        logger.info("%s mode: %d samples", mode, len(ds))
    elif mode == "unified":
        ds_embedding = datasets.concatenate_datasets([
            t for t in train_ds if "query" in t.features
        ])
        ds_generative = datasets.concatenate_datasets([
            t for t in train_ds if "text" in t.features
        ])
        logger.info("Unified mode: %d embedding samples, %d generative samples",
            len(ds_embedding), len(ds_generative)
        )
        for t in train_ds:
            if "query" in t.features:
                num_samples = len(t)
                embedding_dataset_lengths.append(num_samples)
        ds = [ds_embedding, ds_generative]
    else:
        raise NotImplementedError(mode)
    
    return ds, ds_name_to_samples, embedding_dataset_lengths
