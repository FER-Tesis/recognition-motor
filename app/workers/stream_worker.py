import time
from kinesis_client import get_kinesis_client
from config import STREAM_NAME
from utils.frame_decoder import decode_record_data
from emotion_inference import infer_emotion
from backend_client import send_emotion_event
from utils.checkpoint_manager import save_sequence_number, load_sequence_number


def process_shard(shard_id: str, model, kinesis_client):
    print(f"🔁 Iniciando lector para shard {shard_id}")

    last_seq = load_sequence_number(shard_id)

    if last_seq:
        print(f"📌 Checkpoint encontrado para {shard_id}: retomando desde seq {last_seq}")
        iterator_type = "AFTER_SEQUENCE_NUMBER"
        iterator_kwargs = {
            "StartingSequenceNumber": last_seq
        }
    else:
        print(f"📂 No hay checkpoint para {shard_id}. Leyendo desde TRIM_HORIZON.")
        iterator_type = "TRIM_HORIZON"
        iterator_kwargs = {}

    shard_iter_resp = kinesis_client.get_shard_iterator(
        StreamName=STREAM_NAME,
        ShardId=shard_id,
        ShardIteratorType=iterator_type,
        **iterator_kwargs
    )
    shard_iterator = shard_iter_resp["ShardIterator"]

    while True:
        resp = kinesis_client.get_records(
            ShardIterator=shard_iterator,
            Limit=50
        )

        records = resp.get("Records", [])
        shard_iterator = resp.get("NextShardIterator")

        if not records:
            time.sleep(0.4)
            continue

        for record in records:

            seq_num = record["SequenceNumber"]

            try:
                frame_bgr, camera_id, capture_session_id, timestamp = decode_record_data(record["Data"])
                emotion = infer_emotion(model, frame_bgr)

                print(f"🧠 Emoción detectada para cámara {camera_id}: {emotion}")
                
                send_emotion_event(camera_id, capture_session_id, emotion, timestamp)

                save_sequence_number(shard_id, seq_num)

            except Exception as e:
                print(f"❌ Error procesando record en shard {shard_id}: {e}")
