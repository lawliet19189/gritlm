import os
from datasets import load_dataset

# Datasets to download
datasets = [
    "OpenAssistant/oasst1",
    "commonsense_qa",
    "math_qa",
    "web_questions",
    "wiki_qa",
    "yahoo_answers_qa",
    "freebase_qa",
    "ms_marco",
    "pubmed_qa",
    "quarel"
]

# Cache directory
cache_dir = "/u/nlp/data/gritlm/datasets_cache"

# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# Download each dataset
for dataset_name in datasets:
    print(f"Downloading {dataset_name}...")
    try:
        dataset = load_dataset(dataset_name, cache_dir=cache_dir)
        print(f"Downloaded {dataset_name}")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")

print("Download process completed.")