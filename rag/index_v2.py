# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import math
import os
import pickle
import logging
from typing import Optional, Tuple, Union

from tqdm import tqdm
import faiss
import faiss.contrib.torch_utils
import numpy as np
import torch
from rag.utils import *


logger = logging.getLogger(__name__)

EMBEDDINGS_DIM = 512#4096
EMBEDDING_DTYPE = torch.float16

FAISSGPUIndex = Union[
    faiss.GpuIndexIVFFlat,
    faiss.GpuIndexIVFPQ,
    faiss.GpuIndexIVFScalarQuantizer,
    faiss.GpuIndexFlatIP,
]
FAISSIndex = Union[FAISSGPUIndex, faiss.IndexPQ]

GPUIndexConfig = Union[
    faiss.GpuIndexIVFPQConfig,
    faiss.GpuIndexIVFFlatConfig,
    faiss.GpuIndexIVFScalarQuantizerConfig,
    faiss.GpuIndexFlatConfig,
]
BITS_PER_CODE: int = 8
CHUNK_SPLIT: int = 3


def serialize_listdocs(ids):
    ids = pickle.dumps(ids)
    ids = torch.tensor(list(ids), dtype=torch.uint8).cuda()
    return ids


def deserialize_listdocs(ids):
    return [pickle.loads(x.cpu().numpy().tobytes()) for x in ids]


class DistributedIndex(object):
    def __init__(self,
                 dtype=EMBEDDING_DTYPE,
        **kwargs):
        self.embeddings = None
        self.doc_map = dict()
        self.is_in_gpu = True if torch.cuda.is_available() else False
        self.dtype = EMBEDDING_DTYPE
        self.dtype = dtype

    def init_embeddings(self, passages, dim: Optional[int] = EMBEDDINGS_DIM):
        self.doc_map = {i: doc for i, doc in enumerate(passages)}
        self.embeddings = torch.zeros(dim, (len(passages)), dtype=EMBEDDING_DTYPE)
        # if self.is_in_gpu:
        #     self.embeddings = self.embeddings.cuda()

    def _get_saved_embedding_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"embeddings.{shard}.pt")

    def _get_saved_passages_path(self, save_dir: str, shard: int) -> str:
        return os.path.join(save_dir, f"passages.{shard}.pt")

    def save_index(
        self,
        path: str,
        total_saved_shards: int,
        overwrite_saved_passages: bool = False,
    ) -> None:
        """
        Saves index state to disk, which can later be loaded by the load_index method.
        Specifically, it saves the embeddings and passages into total_saved_shards separate file shards.
        This option enables loading the index in another session with a different number of workers, as long as the number of workers is divisible by total_saved_shards.
        Note that the embeddings will always be saved to disk (it will overwrite any embeddings previously saved there).
        The passages will only be saved to disk if they have not already been written to the save directory before, unless the option --overwrite_saved_passages is passed.
        """
        assert self.embeddings is not None
        rank = get_rank()
        ws = get_world_size()
        assert (
            total_saved_shards % ws == 0
        ), f"N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        n_embeddings = self.embeddings.shape[1]
        embeddings_per_shard = math.ceil(n_embeddings / shards_per_worker)
        assert n_embeddings == len(self.doc_map), len(self.doc_map)
        for shard_ind, (shard_start) in enumerate(
            range(0, n_embeddings, embeddings_per_shard)
        ):
            shard_end = min(shard_start + embeddings_per_shard, n_embeddings)
            shard_id = (
                shard_ind + rank * shards_per_worker
            )  # get global shard number
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            if (
                not os.path.exists(passage_shard_path)
                or overwrite_saved_passages
            ):
                passage_shard = [
                    self.doc_map[i] for i in range(shard_start, shard_end)
                ]
                with open(passage_shard_path, "wb") as fobj:
                    pickle.dump(
                        passage_shard, fobj, protocol=pickle.HIGHEST_PROTOCOL
                    )
            embeddings_shard = self.embeddings[:, shard_start:shard_end].clone()
            embedding_shard_path = self._get_saved_embedding_path(
                path, shard_id
            )
            torch.save(embeddings_shard, embedding_shard_path)

    # def load_index(self, path: str, total_saved_shards: int):
    #     """
    #     Loads sharded embeddings and passages files (no index is loaded).
    #     """
    #     rank = get_rank()
    #     ws = get_world_size()
    #     assert (
    #         total_saved_shards % ws == 0
    #     ), f"N workers must be a multiple of shards to save"
    #     shards_per_worker = total_saved_shards // ws
    #     passages = []
    #     embeddings = []
    #     for shard_id in range(
    #         rank * shards_per_worker, (rank + 1) * shards_per_worker
    #     ):
    #         passage_shard_path = self._get_saved_passages_path(path, shard_id)
    #         with open(passage_shard_path, "rb") as fobj:
    #             passages.append(pickle.load(fobj))
    #         embeddings_shard_path = self._get_saved_embedding_path(
    #             path, shard_id
    #         )
    #         embeddings.append(
    #             torch.load(embeddings_shard_path, map_location="cpu").cuda()
    #         )
    #     self.doc_map = {}
    #     n_passages = 0
    #     for chunk in passages:
    #         for p in chunk:
    #             self.doc_map[n_passages] = p
    #             n_passages += 1
    #     self.embeddings = torch.concat(embeddings, dim=1)
    
    def load_index(self, path: str, total_saved_shards: int):
        rank = get_rank()
        ws = get_world_size()
        assert total_saved_shards % ws == 0, "N workers must be a multiple of shards to save"
        shards_per_worker = total_saved_shards // ws
        passages = []
        embeddings = []
        for shard_id in range(rank * shards_per_worker, (rank + 1) * shards_per_worker):
            passage_shard_path = self._get_saved_passages_path(path, shard_id)
            with open(passage_shard_path, "rb") as fobj:
                passages.append(pickle.load(fobj))
            embeddings_shard_path = self._get_saved_embedding_path(path, shard_id)
            # Load embeddings to CPU memory
            embeddings.append(torch.load(embeddings_shard_path, map_location="cpu"))
        
        self.doc_map = {}
        n_passages = 0
        for chunk in passages:
            for p in chunk:
                self.doc_map[n_passages] = p
                n_passages += 1
        # Keep embeddings in CPU memory
        self.embeddings = torch.cat(embeddings, dim=1)
        self.embeddings_on_gpu = False
    
    def to_gpu(self, batch_size=1000):
        """
        Transfers embeddings to GPU in batches.
        """
        if not self.embeddings_on_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gpu_embeddings = []
            for i in range(0, self.embeddings.shape[1], batch_size):
                batch = self.embeddings[:, i:i+batch_size].to(device)
                gpu_embeddings.append(batch)
            self.embeddings = torch.cat(gpu_embeddings, dim=1)
            self.embeddings_on_gpu = True
    
    def _compute_scores_and_indices(self, allqueries: torch.tensor, topk: int) -> Tuple[torch.tensor, torch.tensor]:
        self.ensure_index_ready()
        self.to_gpu()  # Ensure embeddings are on GPU
        scores = torch.matmul(allqueries.half(), self.embeddings)
        scores, indices = torch.topk(scores, topk, dim=1)
        
        return scores, indices

    ####

    # def _compute_scores_and_indices(
    #     self, allqueries: torch.tensor, topk: int
    # ) -> Tuple[torch.tensor, torch.tensor]:
    #     """
    #     Computes the distance matrix for the query embeddings and embeddings chunk and returns the k-nearest neighbours and corresponding scores.
    #     """
    #     scores = torch.matmul(allqueries.half(), self.embeddings)
    #     scores, indices = torch.topk(scores, topk, dim=1)

    #     return scores, indices

    @torch.no_grad()
    def search_knn(self, queries, topk):
        """
        Conducts exhaustive search of the k-nearest neighbours using the inner product metric.
        """
        allqueries = varsize_all_gather(queries)
        allsizes = get_varsize(queries)
        allsizes = np.cumsum([0] + allsizes.cpu().tolist())
        # compute scores for the part of the index located on each process
        scores, indices = self._compute_scores_and_indices(allqueries, topk)
        indices = indices.tolist()
        docs = [
            [self.doc_map[x] for x in sample_indices]
            for sample_indices in indices
        ]
        if torch.distributed.is_initialized():
            docs = [
                docs[allsizes[k] : allsizes[k + 1]]
                for k in range(len(allsizes) - 1)
            ]
            docs = [serialize_listdocs(x) for x in docs]
            scores = [
                scores[allsizes[k] : allsizes[k + 1]]
                for k in range(len(allsizes) - 1)
            ]
            gather_docs = [
                varsize_gather(docs[k], dst=k, dim=0)
                for k in range(get_world_size())
            ]
            gather_scores = [
                varsize_gather(scores[k], dst=k, dim=1)
                for k in range(get_world_size())
            ]
            rank_scores = gather_scores[get_rank()]
            rank_docs = gather_docs[get_rank()]
            scores = torch.cat(rank_scores, dim=1)
            rank_docs = deserialize_listdocs(rank_docs)
            merge_docs = [[] for _ in range(queries.size(0))]
            for docs in rank_docs:
                for k, x in enumerate(docs):
                    merge_docs[k].extend(x)
            docs = merge_docs
        _, subindices = torch.topk(scores, topk, dim=1)
        scores = scores.tolist()
        subindices = subindices.tolist()
        # Extract topk scores and associated ids
        scores = [
            [scores[k][j] for j in idx] for k, idx in enumerate(subindices)
        ]
        docs = [[docs[k][j] for j in idx] for k, idx in enumerate(subindices)]
        return docs, scores

    def is_index_trained(self) -> bool:
        return True
        

class DistributedFAISSIndex(DistributedIndex):
    # def __init__(self, index_type: str, code_size: Optional[int] = None):
    #     super().__init__()
    #     self.embeddings = None
    #     self.doc_map = dict()
    #     self.faiss_gpu_index = None
    #     self.gpu_resources = None
    #     self.faiss_index_trained = False
    #     self.faiss_index_type = index_type
    #     self.code_size = code_size
    #     self.is_in_gpu = False
    
    def __init__(self, index_type: str, code_size: Optional[int] = None):
        super().__init__()
        self.embeddings = None
        self.doc_map = dict()
        self.faiss_gpu_index = None
        self.faiss_index_type = index_type
        self.code_size = code_size
        

    def _get_faiss_index_filename(self, save_index_path: str) -> str:
        """
        Creates the filename to save the trained index to using the index type, code size (if not None) and rank.
        """
        rank = get_rank()
        if self.code_size:
            return (
                save_index_path
                + f"/index{self.faiss_index_type}_{str(self.code_size)}_rank_{rank}.faiss"
            )
        return (
            save_index_path + f"/index{self.faiss_index_type}_rank_{rank}.faiss"
        )

    def _add_embeddings_to_gpu_index(self) -> None:
        """
        Add embeddings to index and sets the nprobe parameter.
        """
        assert (
            self.faiss_gpu_index is not None
        ), "The FAISS GPU index was not correctly instantiated."
        assert (
            self.faiss_gpu_index.is_trained == True
        ), "The FAISS index has not been trained."
        if self.faiss_gpu_index.ntotal == 0:
            self._add_embeddings_by_chunks()

    def _add_embeddings_by_chunks(self) -> None:
        _, num_points = self.embeddings.shape
        chunk_size = num_points // CHUNK_SPLIT
        split_embeddings = [
            self.embeddings[:, 0:chunk_size],
            self.embeddings[:, chunk_size : 2 * chunk_size],
            self.embeddings[:, 2 * chunk_size : num_points],
        ]
        for embeddings_chunk in split_embeddings:
            if isinstance(self.faiss_gpu_index, FAISSGPUIndex.__args__):
                self.faiss_gpu_index.add(
                    self._cast_to_torch32(embeddings_chunk.T)
                )
            else:
                self.faiss_gpu_index.add(
                    self._cast_to_numpy(embeddings_chunk.T)
                )

    def _compute_scores_and_indices(
        self, allqueries: torch.tensor, topk: int
    ) -> Tuple[torch.tensor, torch.tensor]:
        """
        Computes the distance matrix for the query embeddings and embeddings chunk and returns the k-nearest neighbours and corresponding scores.
        """
        _, num_points = self.embeddings.shape
        self.faiss_gpu_index.nprobe = math.floor(math.sqrt(num_points))
        self._add_embeddings_to_gpu_index()
        if isinstance(self.faiss_gpu_index, FAISSGPUIndex.__args__):
            scores, indices = self.faiss_gpu_index.search(
                self._cast_to_torch32(allqueries), topk
            )
        else:
            np_scores, indices = self.faiss_gpu_index.search(
                self._cast_to_numpy(allqueries), topk
            )
            scores = torch.from_numpy(np_scores).cuda()
        return scores.half(), indices

    def save_index(
        self, save_index_path: str, save_index_n_shards: int
    ) -> None:
        """
        Saves the embeddings and passages and if there is a FAISS index, it saves it.
        """
        super().save_index(save_index_path, save_index_n_shards)
        self._save_faiss_index(save_index_path)

    def _save_faiss_index(self, path: str) -> None:
        """
        Moves the GPU FAISS index to CPU and saves it to a .faiss file.
        """
        index_path = self._get_faiss_index_filename(path)
        assert (
            self.faiss_gpu_index is not None
        ), "There is no FAISS index to save."
        cpu_index = faiss.index_gpu_to_cpu(self.faiss_gpu_index)
        faiss.write_index(cpu_index, index_path)

    def _load_faiss_index(self, load_index_path: str) -> None:
        """
        Loads a FAISS index and moves it to the GPU.
        """
        faiss_cpu_index = faiss.read_index(load_index_path)
        # move to GPU
        self._move_index_to_gpu(faiss_cpu_index)
    
    def load_faiss_index(self, faiss_path: str, passages: list[str]):
        """
        Loads a saved FAISS index and initializes doc_map with provided passages.
        """
        print(f"Loading FAISS index from {faiss_path}")
        self.faiss_gpu_index = faiss.read_index(faiss_path)
        
        print("Initializing doc_map with provided passages")
        self.doc_map = {i: doc for i, doc in enumerate(passages)}

        print(f"Loaded index with {self.faiss_gpu_index.ntotal} vectors and {len(self.doc_map)} passages")

        # Sanity check
        if self.faiss_gpu_index.ntotal != len(self.doc_map.values()):
            print(f"Warning: Number of vectors in FAISS index ({self.faiss_gpu_index.ntotal}) "
                  f"does not match number of passages ({len(self.doc_map.values())})")


    # def load_index(self, path: str, total_saved_shards: int) -> None:
    #     """
    #     Loads passage embeddings and passages and a faiss index (if it exists).
    #     Otherwise, it initialises and trains the index in the GPU with GPU FAISS.
    #     """
    #     super().load_index(path, total_saved_shards)
    #     load_index_path = self._get_faiss_index_filename(path)
    #     if os.path.exists(load_index_path):
    #         self._load_faiss_index(load_index_path)
    #     else:
    #         self.train_index()
    
    # def load_index(self, path: str, total_saved_shards: int) -> None:
    #     super().load_index(path, total_saved_shards)
    #     load_index_path = self._get_faiss_index_filename(path)
    #     if os.path.exists(load_index_path):
    #         self._load_faiss_index(load_index_path)
    #     else:
    #         self.index_prepared = False
    #         logger.info("Index not found. It will be trained when first needed.")
    
    def load_index(self, path: str, total_saved_shards: int) -> None:
        """
        Loads passage embeddings and passages. Keeps embeddings in CPU memory.
        """
        super().load_index(path, total_saved_shards)
        # At this point, self.embeddings are in CPU memory

    
    def ensure_index_ready(self):
        if not self.index_prepared:
            self._prepare_index()

    def _prepare_index(self):
        logger.info("Preparing the index...")
        self._initialise_index()
        self.to_gpu()  # Transfer embeddings to GPU before training
        self._train_index_parallel()
        self.index_prepared = True

    def _train_index_parallel(self):
        """
        Trains the index using parallel processing.
        """
        num_processes = min(mp.cpu_count(), 4)  # Use up to 4 processes
        chunk_size = len(self.embeddings) // num_processes

        with mp.Pool(num_processes) as pool:
            results = list(tqdm(
                pool.imap(self._train_index_chunk, range(num_processes)),
                total=num_processes,
                desc="Training index"
            ))

        # Merge results if necessary
        # This step depends on the specific FAISS index type and may need adjustment
        for partial_index in results[1:]:
            self.faiss_gpu_index.merge_from(partial_index, self.faiss_gpu_index.ntotal)

    def _train_index_chunk(self, chunk_id):
        """
        Trains a chunk of the index.
        """
        start = chunk_id * chunk_size
        end = start + chunk_size if chunk_id < num_processes - 1 else len(self.embeddings)
        chunk_embeddings = self.embeddings[:, start:end]

        if isinstance(self.faiss_gpu_index, FAISSGPUIndex.__args__):
            return self.faiss_gpu_index.train(self._cast_to_torch32(chunk_embeddings.T))
        else:
            return self.faiss_gpu_index.train(self._cast_to_numpy(chunk_embeddings.T))

    ##################


    def is_index_trained(self) -> bool:
        if self.faiss_gpu_index is None:
            return self.faiss_index_trained
        return not self.faiss_gpu_index.is_trained

    def _initialise_index(self) -> None:
        """
        Initialises the index in the GPU with GPU FAISS.
        Supported gpu index types: IVFFlat, IndexFlatIP, IndexIVFPQ, IVFSQ.
        """
        dimension, num_points = self.embeddings.shape
        # @TODO: Add support to set the n_list and n_probe parameters.
        n_list = math.floor(math.sqrt(num_points))
        self.faiss_gpu_index = self.gpu_index_factory(dimension, n_list)

    @torch.no_grad()
    def _set_gpu_options(self) -> faiss.GpuMultipleClonerOptions:
        """
        Returns the GPU cloner options neccessary when moving a CPU index to the GPU.
        """
        cloner_opts = faiss.GpuClonerOptions()
        cloner_opts.useFloat16 = True
        cloner_opts.usePrecomputed = False
        cloner_opts.indicesOptions = faiss.INDICES_32_BIT
        return cloner_opts

    @torch.no_grad()
    def _set_index_config_options(
        self, index_config: GPUIndexConfig
    ) -> GPUIndexConfig:
        """
        Returns the GPU config options for GPU indexes.
        """
        index_config.device = torch.cuda.current_device()
        index_config.indicesOptions = faiss.INDICES_32_BIT
        index_config.useFloat16LookupTables = True
        return index_config

    def _create_PQ_index(self, dimension) -> FAISSIndex:
        """
        GPU config options for PQ index
        """
        cpu_index = faiss.index_factory(
            dimension, "PQ" + str(self.code_size), faiss.METRIC_INNER_PRODUCT
        )
        cfg = self._set_gpu_options()
        return faiss.index_cpu_to_gpu(
            self.gpu_resources, self.embeddings.get_device(), cpu_index, cfg
        )

    @torch.no_grad()
    def gpu_index_factory(
        self, dimension: int, n_list: Optional[int] = None
    ) -> FAISSIndex:
        """
        Instantiates and returns the selected GPU index class.
        """
        self.gpu_resources = faiss.StandardGpuResources()
        if self.faiss_index_type == "ivfflat":
            config = self._set_index_config_options(
                faiss.GpuIndexIVFFlatConfig()
            )
            return faiss.GpuIndexIVFFlat(
                self.gpu_resources,
                dimension,
                n_list,
                faiss.METRIC_INNER_PRODUCT,
                config,
            )
        elif self.faiss_index_type == "flat":
            config = self._set_index_config_options(faiss.GpuIndexFlatConfig())
            return faiss.GpuIndexFlatIP(self.gpu_resources, dimension, config)
        elif self.faiss_index_type == "pq":
            return self._create_PQ_index(dimension)
        elif self.faiss_index_type == "ivfpq":
            config = self._set_index_config_options(faiss.GpuIndexIVFPQConfig())
            return faiss.GpuIndexIVFPQ(
                self.gpu_resources,
                dimension,
                n_list,
                self.code_size,
                BITS_PER_CODE,
                faiss.METRIC_INNER_PRODUCT,
                config,
            )
        elif self.faiss_index_type == "ivfsq":
            config = self._set_index_config_options(
                faiss.GpuIndexIVFScalarQuantizerConfig()
            )
            qtype = faiss.ScalarQuantizer.QT_4bit
            return faiss.GpuIndexIVFScalarQuantizer(
                self.gpu_resources,
                dimension,
                n_list,
                qtype,
                faiss.METRIC_INNER_PRODUCT,
                True,
                config,
            )
        else:
            raise ValueError("unsupported index type")

    # @torch.no_grad()
    # def train_index(self) -> None:
    #     """
    #     It initialises the index and trains it according to the refresh index schedule.
    #     """
    #     if self.faiss_gpu_index is None:
    #         self._initialise_index()
    #     self.faiss_gpu_index.reset()
    #     if isinstance(self.faiss_gpu_index, FAISSGPUIndex.__args__):
    #         logger.info("Training with FAISS as torch32")
    #         self.faiss_gpu_index.train(self._cast_to_torch32(self.embeddings.T))
    #     else:
    #         logger.info("Training with FAISS as numpy")
    #         self.faiss_gpu_index.train(self._cast_to_numpy(self.embeddings.T))

    def train_index(self, save_path: str = None) -> None:
        """
        Trains the FAISS index using CPU embeddings and optionally saves it.
        """
        if self.faiss_gpu_index is None:
            self._initialise_index()

        print("Training FAISS index...")
        self.faiss_gpu_index.train(self._cast_to_numpy(self.embeddings.T))
        
        self._add_embeddings_to_index()

        if save_path:
            self.save_faiss_index(save_path)

    @torch.no_grad()
    def _cast_to_torch32(self, embeddings: torch.tensor) -> torch.tensor:
        """
        Converts a torch tensor to a contiguous float 32 torch tensor.
        """
        return embeddings.type(torch.float32).contiguous()

    @torch.no_grad()
    def _cast_to_numpy(self, embeddings: torch.tensor) -> np.ndarray:
        """
        Converts a torch tensor to a contiguous numpy float 32 ndarray.
        """
        return (
            embeddings.cpu()
            .to(dtype=torch.float16)
            .numpy()
            .astype("float32")
            .copy(order="C")
        )

    @torch.no_grad()
    def _move_index_to_gpu(self, cpu_index: FAISSIndex) -> None:
        """
        Moves a loaded index to GPU.
        """
        self.gpu_resources = faiss.StandardGpuResources()
        cfg = self._set_gpu_options()
        self.faiss_gpu_index = faiss.index_cpu_to_gpu(
            self.gpu_resources, torch.cuda.current_device(), cpu_index, cfg
        )

    def _initialise_index(self) -> None:
        """
        Initialises the FAISS index on CPU.
        """
        dimension, num_points = self.embeddings.shape
        n_list = min(EMBEDDINGS_DIM, math.floor(math.sqrt(num_points)))
        
        if self.faiss_index_type == "ivfflat":
            self.faiss_gpu_index = faiss.IndexIVFFlat(
                faiss.IndexFlatIP(dimension), dimension, n_list, faiss.METRIC_INNER_PRODUCT
            )
        elif self.faiss_index_type == "ivfpq":
            self.faiss_gpu_index = faiss.IndexIVFPQ(
                faiss.IndexFlatIP(dimension), dimension, n_list, 
                self.code_size, BITS_PER_CODE, faiss.METRIC_INNER_PRODUCT
            )
        else:
            raise ValueError(f"Unsupported index type: {self.faiss_index_type}")

    def _add_embeddings_to_index(self) -> None:
        """
        Add embeddings to the trained index.
        """
        print("Adding embeddings to index...")
        self.faiss_gpu_index.add(self._cast_to_numpy(self.embeddings.T))

    def save_faiss_index(self, save_path: str) -> None:
        """
        Saves the FAISS index to a file.
        """
        if self.faiss_gpu_index is None:
            raise ValueError("No index to save. Please train the index first.")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving FAISS index to {save_path}")
        faiss.write_index(self.faiss_gpu_index, save_path)

    @staticmethod
    def _cast_to_numpy(embeddings: torch.Tensor) -> np.ndarray:
        """
        Converts a torch tensor to a contiguous numpy float32 ndarray.
        """
        return embeddings.cpu().numpy().astype('float32').copy(order='C')

    @torch.no_grad()
    def search_knn(self, queries: torch.Tensor, k: int) -> Tuple[list[list[str]], np.ndarray]: # TODO: either float or np.ndarray
        """
        Performs k-nearest neighbor search using the loaded FAISS index.
        """
        if self.faiss_gpu_index is None:
            raise ValueError("FAISS index not loaded. Call load_faiss_index first.")

        # Convert queries to numpy array
        if isinstance(queries, torch.Tensor):
            queries = queries.cpu().numpy()
        
        query_numpy = queries.astype('float32')

        # Perform the search
        scores, indices = self.faiss_gpu_index.search(query_numpy, k)

        # Fetch the corresponding passages
        results = []
        for query_indices in indices:
            query_results = [self.doc_map[int(idx)] for idx in query_indices if int(idx) in self.doc_map]
            results.append(query_results)

        return results, scores
    
    @staticmethod
    def save_passages(passages: dict, save_path: str):
        """
        Saves passages to a file.
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"Saving passages to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(passages, f)