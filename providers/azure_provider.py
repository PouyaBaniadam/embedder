import asyncio
from typing import Dict, Any
from azure.ai.inference import EmbeddingsClient
from azure.core.credentials import AzureKeyCredential
from base import BaseEmbeddingProvider
from logger_config import logger

class AzureEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, api_key: str, endpoint: str, model_name: str):
        self.client = EmbeddingsClient(endpoint=endpoint, credential=AzureKeyCredential(api_key))
        self.model_name = model_name

    async def generate_embedding(self, text: str) -> Dict[str, Any]:
        try:
            response = await asyncio.to_thread(self.client.embed, input=[text], model=self.model_name)
            item = response.data[0]
            return {
                "embedding": item.embedding,
                "index": item.index,
                "usage": {"prompt_tokens": response.usage.prompt_tokens, "total_tokens": response.usage.total_tokens}
            }
        except Exception as e:
            logger.error(f"Azure Embedding Error: {str(e)}")
            raise e