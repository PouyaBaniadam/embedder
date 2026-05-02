import pika
import json

RABBITMQ_HOST = "192.168.200.165"
RABBITMQ_PORT = 18011
QUEUE_OUTPUT = "embedding_output_queue"


def callback(ch, method, properties, body):
    response = json.loads(body.decode())

    print("\n" + "=" * 50)
    print(" [x] Received Processed Result:")
    print(f"     ID: {response.get('id')}")
    print(f"     Status: {response.get('status')}")

    if response.get('status') == 'success':
        embedding_array = response['data']['embedding']
        print(f"     Embedding Length: {len(embedding_array)}")
        print(f"     Embedding Preview: {embedding_array[:5]} ...")
        print(f"     Tokens Used: {response['data']['usage']}")
    else:
        print(f"     Error: {response.get('error')}")

    print("=" * 50 + "\n")


def start_receiving():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST, port=RABBITMQ_PORT)
    )
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_OUTPUT)

    print(f" [*] Waiting for results in '{QUEUE_OUTPUT}'. To exit press CTRL+C")

    channel.basic_consume(
        queue=QUEUE_OUTPUT,
        on_message_callback=callback,
        auto_ack=True
    )

    channel.start_consuming()


if __name__ == "__main__":
    start_receiving()
