from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        pass