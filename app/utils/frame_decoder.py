import base64
import json
import numpy as np
import cv2

def decode_record_data(record_data: bytes):
    json_str = record_data.decode("utf-8")
    payload = json.loads(json_str)

    camera_id = payload["camera_id"]
    capture_session_id = payload["capture_session_id"]
    timestamp = payload["timestamp"]
    frame_b64 = payload["frame"]

    jpg_bytes = base64.b64decode(frame_b64)
    np_arr = np.frombuffer(jpg_bytes, np.uint8)
    frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        raise ValueError("No se pudo decodificar el frame JPEG")

    return frame_bgr, camera_id, capture_session_id, timestamp
