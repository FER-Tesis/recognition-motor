import threading
import time

from kinesis_client import get_kinesis_client
from config import STREAM_NAME
from model_loader import load_model
from workers.stream_worker import process_shard

def main():
    model = load_model()

    kinesis_client = get_kinesis_client()

    desc = kinesis_client.describe_stream(StreamName=STREAM_NAME)
    shards = desc["StreamDescription"]["Shards"]

    print("========== WORKER IA ==========")
    print(f"Stream: {STREAM_NAME}")
    print(f"Shards detectados: {len(shards)}")
    for shard in shards:
        print(f" - {shard['ShardId']}")
    print("--------------------------------")

    for shard in shards:
        shard_id = shard["ShardId"]
        t = threading.Thread(
            target=process_shard,
            args=(shard_id, model, kinesis_client),
            daemon=True
        )
        t.start()

    print("🧠 Worker IA iniciado. CTRL+C para detener.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋Worker IA detenido.")

if __name__ == "__main__":
    main()
