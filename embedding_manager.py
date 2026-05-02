import os
from dotenv import load_dotenv
from providers.azure_provider import AzureEmbeddingProvider
from providers.hf_provider import HuggingFaceProvider

load_dotenv()

MODEL_MAPPING = {
    "text-embedding-3-large": "azure",
    "text-embedding-3-small": "azure",
    "google/embeddinggemma-300M": "local_hf",
}

def get_embedding_provider(model_name: str):
    provider_name = MODEL_MAPPING.get(model_name)
    if provider_name == "azure":
        return AzureEmbeddingProvider(os.getenv("AZURE_API_KEY"), "https://models.github.ai/inference", model_name)
    elif provider_name == "local_hf":
        return HuggingFaceProvider(model_name=model_name)
    raise ValueError(f"Unsupported model: {model_name}")

async def process_text_to_embedding(text: str, model_name: str) -> dict:
    provider = get_embedding_provider(model_name)
    return await provider.generate_embedding(text)
