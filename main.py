from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import uuid
from datetime import datetime
from embedding_manager import get_embedding_provider, MODEL_MAPPING
from logger_config import logger

app = FastAPI(title="Azure/GitHub Embedding API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingRequest(BaseModel):
    text: str


@app.post("/embed/")
async def create_embedding(
        request: EmbeddingRequest,
        model: str = Query("text-embedding-3-large")
):
    text = request.text

    req_id = str(uuid.uuid4())
    logger.info(f"[{req_id}] Request | Model: {model}")

    if not text.strip():
        return {"id": req_id, "error": "Text is empty", "status": "failed"}

    try:
        service = get_embedding_provider(model)
        result = await service.generate_embedding(text)

        logger.info(f"[{req_id}] Embedding successful.")

        return {
            "id": req_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model_used": model,
            "data": {
                "embedding": result["embedding"],
                "dimensions": len(result["embedding"]),
                "index": result["index"]
            },
            "usage": result["usage"],
            "status": "success"
        }

    except ValueError as ve:
        logger.warning(f"[{req_id}] Validation Error: {str(ve)}")
        return {"id": req_id, "error": str(ve), "status": "failed"}
    except Exception as e:
        logger.critical(f"[{req_id}] System Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.get("/models")
async def list_models():
    return {"supported_models": list(MODEL_MAPPING.keys())}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)