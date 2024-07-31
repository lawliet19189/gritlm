## RAG with GRIT

This document details how to reproduce the RAG experiments in the paper. The result files are also contained in the results folder: https://huggingface.co/datasets/GritLM/results/tree/main/GritLM-7B

### Setup

#### Packages

To setup packages etc. follow the instructions of the main README.md.

#### Index

You don't need the index for latency benchmarking but do need it for performance benchmarking.
We have uploaded the index of GritLM-7B here: https://huggingface.co/datasets/GritLM/index
You can also follow the below to recreate it:

```bash
cd gritlm
python -m rag.prepare_qa --output_directory rag/
wget -O - https://huggingface.co/datasets/BeIR/nq/resolve/main/corpus.jsonl.gz | gunzip > corpus.jsonl
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --passages corpus.jsonl --save_index_path index_nq
```

### Evaluation Data

To run evaluation on Natural Questions, TriviaQA, or MMLU, it is necessary to download and preprocess the data. Run the following to accomplish this:

#### NaturalQuestions & TriviaQA (Ignore if you had built the index previously, since this step would be redundant):

```base
python -m rag.prepare_qa --output_directory rag/
```

#### MMLU:

```base
python -m rag.prepare_mmlu --output_directory rag/ && mv rag/data/mmlu_data rag/mmlu_data && rm -rf rag/data rag/mmlu_data/data.tar
```

Note: The test data for the respective RAG datasets are as follows:

- NaturalQuestions: `rag/nq_data/test.jsonl`
- TriviaQA: `rag/triviaqa_data/test.jsonl`
- MMLU: `rag/mmlu_data/5-shot/combined_test.jsonl`

### Benchmarking

To run the latency benchmark, do `bash scripts/raglatency.sh` after adjusting the script to your cluster / paths. For performance benchmarking, you can adapt & run the scripts below:

No retrieval:

```bash
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --no_retrieval --load_index_path index_nq --cache query
```

Query then document prompt RAG:

```bash
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --passages corpus.jsonl --load_index_path index_nq --prompt query
```

Query Caching

```bash
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --passages corpus.jsonl --load_index_path index_nq --cache query
```

Query-Doc Caching

```bash
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --passages corpus.jsonl --load_index_path index_nq --cache querydoc
```

Document then query prompt RAG:

```bash
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --passages corpus.jsonl --load_index_path index_nq --prompt doc
```

Doc Caching:

```bash
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --passages corpus.jsonl --load_index_path index_nq --cache doc
```

Doc-Query Caching

```bash
python -m rag.eval --model_name_or_path GritLM/gritlm-7b --eval_data rag/nq_data/test.jsonl --passages corpus.jsonl --load_index_path index_nq --cache docquery
```

### Acknowledgements

The code is adapted from [ATLAS](https://github.com/facebookresearch/atlas).
