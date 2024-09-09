from argparse import ArgumentParser
import json
import os
import random
import re
from datasets import load_dataset
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--corpus_split", type=float, default=0.95, help="Replug uses both text corpus and instructions dataset. This argument is the split for the corpus. The rest will be used for instructions. Suggested value is 0.95")
    parser.add_argument("--max_train_size", type=int, default=None, help="Max number of training samples to use. If None, all training samples will be used.")
    parser.add_argument("--hf_cache_dir", type=str, default="/u/nlp/data/gritlm/datasets_cache", help="Cache directory for Hugging Face datasets")
    parser.add_argument("--corpus_data_path", type=str, default="/u/nlp/data/gritlm/replug/arena-wiki-2024.jsonl", help="The path to the textual corpus data in jsonl format")
    parser.add_argument("--output_path", type=str, required=True)
    return parser.parse_args()

datasets = [
    ("OpenAssistant/oasst1", None),
    ("commonsense_qa", None),
    ("math_qa", None),
    ("web_questions", None),
    ("wiki_qa", None),
    ("yahoo_answers_qa", None),
    ("freebase_qa", None),
    ("ms_marco", "v2.1"),
    ("pubmed_qa", "pqa_artificial"),
    # ("quarel", None)  # TODO: Add this dataset
]

def load_dataset_from_hf(hf_dataset_id: str, hf_dataset_version: str, hf_cache_dir: str):
    dataset = load_dataset(hf_dataset_id, hf_dataset_version, cache_dir=hf_cache_dir)
    return dataset

def download_and_cache_dataset(hf_dataset_id: str, hf_dataset_version: str, hf_cache_dir: str):
    dataset = load_dataset_from_hf(hf_dataset_id, hf_dataset_version, hf_cache_dir)
    dataset_size = len(dataset["train"])
    return dataset_size

def estimate_corpus_size(corpus_data_path: str):
    with open(corpus_data_path, "r") as f:
        for i, line in enumerate(f):
            pass
    return i + 1

def preprocess_and_filter_oasst1(dataset, total_samples: int = 500):
    # pre-map few mappgin to help with preprocessing
    message_id_to_idx_map = {}
    parent_id_to_idx_map = {}
    for idx, item in enumerate(dataset['train']):
        parent_id = item['parent_id']
        message_id = item['message_id']
        
        
        message_id_to_idx_map[message_id] = idx
        parent_id_to_idx_map[parent_id] = idx
        

    
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        message_id = item['message_id']
        parent_id = item['parent_id']
        role = item["role"]
        text = item['text']
    
        x, y = "", ""

        # Ignore child or nested messages
        if parent_id != None:
            continue

        child_idx = parent_id_to_idx_map[message_id]
        x = text
        y = dataset['train'][child_idx]['text']
        
        samples.append((x, y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_quarel_qa(dataset, total_samples: int = 500):
    raise NotImplementedError("Not implemented yet")

def preprocess_and_filter_pubmed_qa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['question']
        y = f"{item['final_decision']}. {item['long_answer']}"

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_msmarco_qa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['query']
        y = item['answers'][0]

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_freebase_qa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['ProcessedQuestion']
        y = item['Parses']['Answers'][0]['AnswersName'][0][0]

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_yahoo_answers_qa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['question']
        y = item['answer']

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_web_qa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['question']
        y = item['answers'][0]

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_wiki_qa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['question']
        y = item['answer']

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_mqa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['Problem']
        options = item['options']
        correct_option = item['correct']
        y = [o.split(f"{correct_option} ) ")[-1] for o in options.split(" , ") if correct_option in o][0]

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

def preprocess_and_filter_cqa(dataset, total_samples: int = 500):
    samples = []
    for idx, item in enumerate(dataset['train'].shuffle(seed=42)):
        x = item['question']
        label_str = item['answerKey']
        label_idx = item["choices"]["label"].index(label_str)
        y = item["choices"]['text'][label_idx]

        samples.append((x,y))

        if len(samples) == total_samples:
            break
    return samples

datasets_config = {
    "OpenAssistant/oasst1": {
        "preprocessing_func": preprocess_and_filter_oasst1,
        "total_samples": 31598
    },
    "commonsense_qa": {
        "preprocessing_func": preprocess_and_filter_cqa,
        "total_samples": 9741
    },
    "math_qa": {
        "preprocessing_func": preprocess_and_filter_mqa,
        "total_samples": 29837
    },
    "web_questions": {
        "preprocessing_func": preprocess_and_filter_web_qa,
        "total_samples": 3778
    },
    "wiki_qa": {
        "preprocessing_func": preprocess_and_filter_wiki_qa,
        "total_samples": 20360
    },
    "quarel": {
        "preprocessing_func": preprocess_and_filter_quarel_qa,
        "total_samples": 1941
    },
    "pubmed_qa": {
        "preprocessing_func": preprocess_and_filter_pubmed_qa,
        "total_samples": 1000
    },
    "ms_marco": {
        "preprocessing_func": preprocess_and_filter_msmarco_qa,
        "total_samples": 80143
    },
    "freebase_qa": {
        "preprocessing_func": preprocess_and_filter_freebase_qa,
        "total_samples": 20358
    },
    "yahoo_answers_qa": {
        "preprocessing_func": preprocess_and_filter_yahoo_answers_qa,
        "total_samples": 87362
    },
    "c_qa": {
        "preprocessing_func": preprocess_and_filter_cqa,
        "total_samples": 500
    }
}

def prepare_instruction_dataset(instruction_dataset_size: int, hf_cache_dir: str):
    instruction_dataset = []
    for dataset_id, dataset_version in tqdm(datasets):
        dataset_config = datasets_config[dataset_id]
        dataset = load_dataset_from_hf(dataset_id, dataset_version, hf_cache_dir)
        instruction_dataset.extend(dataset_config["preprocessing_func"](dataset, dataset_config["total_samples"]))
    
    random.shuffle(instruction_dataset)

    instruction_dataset = instruction_dataset[:instruction_dataset_size]

    return instruction_dataset

def split_text(text: str) -> tuple[str, str]:
    words = text.split()
    half_idx = len(words) // 2
    return " ".join(words[:half_idx]), " ".join(words[half_idx:])

def prepare_corpus_dataset(corpus_data_path: str, corpus_size: int, shuffle: bool = False):
    assert shuffle == False, "Shuffling is not yet implemented since we need to load the entire corpus in memory"

    lines = []
    with open(corpus_data_path, "r") as f:
        line = f.readline()

        while line and len(lines) < corpus_size:
            line = json.loads(f.readline())

            # tokenize the line and split it by half
            first_half, second_half = split_text(line['text'])
            lines.append((first_half, second_half))
    return lines
        



def prepare_replug_data(args):
    os.makedirs(args.hf_cache_dir, exist_ok=True)

    logger.info("Downloading instructions datasets...")
    dataset_size = 0 # used for estimating the size of the dataset
    for dataset_id, dataset_version in tqdm(datasets):
        dataset_size += download_and_cache_dataset(dataset_id, dataset_version, args.hf_cache_dir)
    logger.info("Instructions datasets loaded.")

    logger.info("Estimating corpus size...")
    corpus_size = estimate_corpus_size(args.corpus_data_path)
    logger.info("Corpus size estimated.")

    logger.info("Estimating total size...")
    total_size = dataset_size + corpus_size
    logger.info("Total size estimated. %d", total_size)

    corpus_split = args.corpus_split
    instruction_split = 1 - corpus_split
    max_corpus_size = int(args.max_train_size * corpus_split)
    max_instruction_size = int(args.max_train_size * instruction_split)


    logger.info("Preparing dataset...")
    instruction_dataset = prepare_instruction_dataset(max_instruction_size, args.hf_cache_dir)
    corpus_dataset = prepare_corpus_dataset(args.corpus_data_path, max_corpus_size)

    # merge the two datasets
    merged_dataset = corpus_dataset + instruction_dataset

    # save the dataset
    with open(args.output_path, "w") as f:
        for x, y in merged_dataset:
            data = {"text": [x, y]}
            json.dump(data, f)
            f.write("\n")

if __name__ == "__main__":
    args = parse_args()
    prepare_replug_data(args)