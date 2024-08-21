import argparse
import logging
import os

from rag.eval import build_index
import torch
from rag.index import DistributedIndex#, load_or_initialize_index
from rag.index_io import load_or_initialize_index
from gritlm import GritLM


logger = logging.getLogger(__name__)
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        default="gritlm",
        type=str
    )
    parser.add_argument(
        "--use_file_passages",
        type=bool,
        default=False,
        help="Whether to use file passages or not"
    )
    parser.add_argument(
        "--passages",
        nargs="+",
        help="list of paths to jsonl files containing passages to index and retrieve from."
    )
    parser.add_argument(
        "--max_passages",
        type=int,
        default=-1,
        help="limit number of passages to index",
    )
    parser.add_argument(
        "--limit_start",
        type=int,
        default=0,
        help="denote start of limit"
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="None / query / doc / querydoc / docquery",
    )
    parser.add_argument(
        "--save_index_n_shards",
        default=1,
        type=int,
        help="how many shards to save an index to file with. Must be an integer multiple of the number of workers.",
    )
    parser.add_argument(
        "--save_index_path",
        default=None,
        type=str,
        help="path for saving the index and/or embeddings",
    )
    parser.add_argument(
        "--per_gpu_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU.",
    )
    parser.add_argument(
        "--embedbs",
        default=128,
        type=int,
        help="Batch size for embedding docs.",
    )
    parser.add_argument(
        "--index_mode",
        default="faiss",
        type=str,
        help="Indexing mode to use. Either flat or faiss."
    )
    parser.add_argument(
        "--embed_max_vectors_in_gpu",
        default=100000,
        type=int
    )
    parser.add_argument(
        "--move_cache_to_cpu",
        action="store_true",
        help="Move doc cache to cpu"
    )
    parser.add_argument(
        "--idxdtype", default="float32", type=str, help="Index dtype"
    )
    parser.add_argument(
        "--pooling_method", default="mean", type=str, help="GritLM Attn mode"
    )
    parser.add_argument(
        "--attn", default="bbcc", type=str, help="GritLM Attn mode"
    )
    parser.add_argument(
        "--faiss_code_size", default=16, type=int, help="Faiss code size."
    )
    parser.add_argument(
        "--attn_implementation",
        default="sdpa",
        type=str,
        help="GritLM Attn imp",
    )
    parser.add_argument(
        "--faiss_index_type",
        default="ivfpq",
        type=str,
        help="Faiss index type."
    )
    return parser.parse_args()
    
def custom_jsonl_transformer(item):
    if 'title' in item and 'section' in item and 'text' in item:
        item['text'] = f"{item['title']}: {item['section']}\n{item['text']}"
    elif 'title' in item and 'section' in item:
        item['text'] = f"{item['title']}: {item['section']}"
    elif 'title' in item and 'text' in item:
        item['text'] = f"{item['title']}\n{item['text']}"
    elif 'section' in item and 'text' in item:
        item['text'] = f"{item['section']}\n{item['text']}"
    elif 'title' in item:
        item['text'] = item['title']
    elif 'section' in item:
        item['text'] = item['section']
    else:
        item['text'] = item.get('text', '')

    # Remove other keys
    keys_to_remove = [key for key in item if key != 'text']
    for key in keys_to_remove:
        del item[key]

    return item

def main():
    args = get_args()
    # Not required for indexing
    args.load_index_path = None
    args.customd = None
    ### 

    logger.info("Initializing index!")
    index, passages = load_or_initialize_index(
        args,
        custom_jsonl_transformer=custom_jsonl_transformer
    )
    logger.info("Total passages retrieved: %s", len(passages))
    
    # TODO: move all of these + model related args to a centralised file and remove redundant code
    gritlm_kwargs = {
        "mode": "embedding",
        "pooling_method": args.pooling_method,
        "attn": args.attn,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": torch.bfloat16,
    }
    model = GritLM(args.model_name_or_path, **gritlm_kwargs)
    model.eval()
    
    if not (model.tokenizer.pad_token) and model.tokenizer.bos_token:
        model.tokenizer.pad_token = model.tokenizer.bos_token
        logger.info("Set pad token to bos token: %s", model.tokenizer.pad_token)
    
    build_index(
        model,
        index,
        passages,
        gpu_embedder_batch_size=args.embedbs,
        cache=args.cache is not None and ("doc" in args.cache),
        move_cache_to_cpu=args.move_cache_to_cpu,
    )
    
    if args.save_index_path is None:
        logger.info("save_index_path not provided. Defaulting to index/index.faiss")
        args.save_index_path = "index/index.faiss"
    os.makedirs(args.save_index_path, exist_ok=True)
    logger.info("Saving final computed index!")
    index.save_index(args.save_index_path, args.save_index_n_shards)
    
if __name__ == "__main__":
    main()