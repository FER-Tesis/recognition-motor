import os
import json

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "checkpoints")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def _get_checkpoint_path(shard_id: str):
    return os.path.join(CHECKPOINT_DIR, f"{shard_id}.json")


def save_sequence_number(shard_id: str, sequence_number: str):
    path = _get_checkpoint_path(shard_id)

    with open(path, "w") as f:
        json.dump({"sequence_number": sequence_number}, f)

    # print(f"[Checkpoint] Guardado shard={shard_id} seq={sequence_number}")


def load_sequence_number(shard_id: str):
    path = _get_checkpoint_path(shard_id)

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("sequence_number")
    except Exception:
        return None
