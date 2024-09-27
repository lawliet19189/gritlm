import torch

from rag.index_io import load_passages
from rag.index_v2 import DistributedFAISSIndex
from rag.passage_encoder import custom_jsonl_transformer
try:
    from transformers import BitsAndBytesConfig
except ImportError:
    BitsAndBytesConfig = None

def get_quantization_config(qlora_enabled: bool):
    if not qlora_enabled:
        return None, False
    
    if BitsAndBytesConfig is None:
        assert qlora_enabled, "BitsAndBytesConfig is not available. Please install it to use 4-bit quantization."
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
def get_rag_train_index(index_path: str, index_passages_path: str, index_passages_num: int, index_type: str, code_size: int):
    index = DistributedFAISSIndex(index_type, code_size)
    all_passages = load_passages(index_passages_path, index_passages_num, custom_jsonl_transformer)
    index.load_faiss_index(index_path, all_passages)

    # free memory
    del all_passages
    return index