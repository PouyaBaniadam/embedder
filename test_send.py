import pika
import json
import uuid

RABBITMQ_HOST = "192.168.200.165"
RABBITMQ_PORT = 18011
QUEUE_INPUT = "embedding_input_queue"


def send_test_message():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
    )
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_INPUT)

    data = {
        "uuid": str(uuid.uuid4()),
        "text": "سلام، این یک متن تستی برای تبدیل به وکتور (Embedding) است.",
        "model_name": "google/embeddinggemma-300M"
    }

    channel.basic_publish(
        exchange="",
        routing_key=QUEUE_INPUT,
        body=json.dumps(data)
    )

    print(f" [x] Message Sent Successfully!")
    print(f"     UUID: {data['uuid']}")
    print(f"     Model: {data['model_name']}")

    connection.close()


if __name__ == "__main__":
    send_test_message()