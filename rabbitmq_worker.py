import os
import json
import uuid
import pika
import asyncio
import threading
from datetime import datetime
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv

from logger_config import logger
from embedding_manager import process_text_to_embedding

load_dotenv()

# --- RabbitMQ Configuration ---
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", 5672))
QUEUE_INPUT = "embedding_input_queue"
QUEUE_OUTPUT = "embedding_output_queue"


def process_message(ch, method, properties, body):
    try:
        # 1. Parse Input Arguments
        data = json.loads(body.decode())
        text = data.get("text")
        req_id = data.get("uuid", str(uuid.uuid4()))
        model_name = data.get("model_name", "text-embedding-3-small")

        logger.info(f"Job Received | ID: {req_id} | Model: {model_name}")

        if not text:
            raise ValueError("Text field is missing or empty.")

        # 2. Process
        embedding_data = asyncio.run(process_text_to_embedding(
            text=text,
            model_name=model_name
        ))

        # 3. Prepare Success Output
        result = {
            "id": req_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "text": text,
            "model_name": model_name,
            "data": embedding_data,
            "status": "success"
        }

    except Exception as e:
        logger.error(f"Error on request {req_id}: {str(e)}")
        result = {
            "id": req_id,
            "error": str(e),
            "status": "failed"
        }

    finally:
        # 4. Publish result
        ch.basic_publish(
            exchange="",
            routing_key=QUEUE_OUTPUT,
            body=json.dumps(result)
        )
        ch.basic_ack(delivery_tag=method.delivery_tag)
        logger.info(f"Job Finished | ID: {req_id}")


def rabbitmq_worker_thread():
    try:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT, heartbeat=0)
        )
        channel = connection.channel()
        channel.queue_declare(queue=QUEUE_INPUT)
        channel.queue_declare(queue=QUEUE_OUTPUT)
        channel.basic_qos(prefetch_count=1)
        channel.basic_consume(queue=QUEUE_INPUT, on_message_callback=process_message)

        logger.info(f" [*] RabbitMQ Worker connected. Listening on {QUEUE_INPUT}")
        channel.start_consuming()
    except Exception as e:
        logger.error(f"RabbitMQ Worker crashed: {str(e)}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    worker_thread = threading.Thread(target=rabbitmq_worker_thread, daemon=True)
    worker_thread.start()
    yield


app = FastAPI(title="Embedding Worker Service", lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run("rabbitmq_worker:app", host="127.0.0.1", port=8000)
