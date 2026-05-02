import os
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, Any
from base import BaseEmbeddingProvider
from logger_config import logger


class HuggingFaceProvider(BaseEmbeddingProvider):
    _instances = {}

    def __new__(cls, model_name: str, *args, **kwargs):
        if model_name not in cls._instances:
            cls._instances[model_name] = super(HuggingFaceProvider, cls).__new__(cls)
        return cls._instances[model_name]

    def __init__(self, model_name: str):
        if hasattr(self, 'initialized'): return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, token=os.getenv("HF_TOKEN")).to(self.device)
        self.model_name = model_name
        self.initialized = True

    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        embeddings = self.model.encode([text], prompt_name="STS", normalize_embeddings=True)
        token_count = len(self.model.tokenizer([text])['input_ids'][0])
        return {
            "embedding": embeddings[0].tolist(),
            "index": 0,
            "usage": {"prompt_tokens": token_count, "total_tokens": token_count}
        }