import httpx
from config import BACKEND_BASE_URL, EMOTION_EVENT_ENDPOINT

def send_emotion_event(camera_id: str, capture_session_id: str, emotion: str, timestamp: str):
    url = BACKEND_BASE_URL + EMOTION_EVENT_ENDPOINT

    payload = {
        "camera_id": camera_id,
        "capture_session_id": capture_session_id,
        "emotion": emotion,
        "timestamp": timestamp,
    }

    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(url, json=payload)

        if resp.status_code == 201:
            print(f"📨 Evento enviado al backend: {payload}")
        else:
            print(f"⚠️ Backend respondió {resp.status_code}: {resp.text}")

    except httpx.RequestError as e:
        print(f"❌ Error al conectar con el backend: {e}")
