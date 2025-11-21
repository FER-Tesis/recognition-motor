import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_FOLDER = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH = os.path.join(MODELS_FOLDER, "effb3_best.pth")

STREAM_NAME = "emotion-frame-stream"
REGION_NAME = "us-east-1"
AWS_PROFILE = "emotion-system"

BACKEND_BASE_URL = "http://localhost:8004"
EMOTION_EVENT_ENDPOINT = "/api/emotion/emotion-events"
