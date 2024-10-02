from .bge_m3 import BGEM3FlagModel
from typing import List
from llama_index.vector_stores.milvus.utils import BaseSparseEmbeddingFunction
import torch


class SparseEmbedding(BaseSparseEmbeddingFunction):
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            device = self._get_device()
            print(f"Using device: {device}")
            self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device=device)
        return self._model

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        elif torch.backends.xla.is_available():
            return "xla"
        else:
            return "cpu"

    def encode_queries(self, queries: List[str]):
        outputs = self.model.encode(
            queries,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def encode_documents(self, documents: List[str]):
        outputs = self.model.encode(
            documents,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )["lexical_weights"]
        return [self._to_standard_dict(output) for output in outputs]

    def _to_standard_dict(self, raw_output):
        result = {}
        for k in raw_output:
            result[int(k)] = raw_output[k]
        return result
