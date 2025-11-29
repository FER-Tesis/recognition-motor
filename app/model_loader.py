import torch
from torchvision import models, transforms as T

from config import MODEL_PATH

IDX_TO_LABEL = ["neutral", "happy", "sad", "surprise", "fear", "disgust", "anger"]

CALIBRATION_WEIGHTS = torch.tensor([
    0.60,  # neutral
    1.00,  # happy
    1.45,  # sad
    1.35,  # surprise
    1.40,  # fear
    1.55,  # disgust
    1.10,  # anger
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((300, 300)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def load_model():
    model = models.efficientnet_b3(weights=None)
    in_feats = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_feats, len(IDX_TO_LABEL))

    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)

    print("✅ Modelo EfficientNet-B3 cargado correctamente.")
    return model

def calibrate_probs(probs: torch.Tensor) -> torch.Tensor:
    w = CALIBRATION_WEIGHTS.to(probs.device)
    adjusted = probs * w
    adjusted = adjusted / adjusted.sum()
    return adjusted

def decision_rule(probs):
    neutral = probs[0].item()
    others = probs[1:]

    max_idx = torch.argmax(probs).item()
    max_other = others.max().item()
    max_other_idx = torch.argmax(others).item()

    disgust_prob = probs[5].item()

    DISGUST_THRESHOLD = 0.17

    if neutral >= 0.75:
        return 0

    if max_other >= 0.20:
        if max_other_idx == 4 and disgust_prob < DISGUST_THRESHOLD:
            filtered = others.clone()
            filtered[4] = -1
            second_best = filtered.max().item()
            second_idx = torch.argmax(filtered).item()

            if second_best >= 0.20:
                return 1 + second_idx
            return 0

        return 1 + max_other_idx

    if max_other >= 0.12:
        if max_other_idx == 4 and disgust_prob < DISGUST_THRESHOLD:
            filtered = others.clone()
            filtered[4] = -1
            second_best = filtered.max().item()
            second_idx = torch.argmax(filtered).item()

            if second_best >= 0.12:
                return 1 + second_idx
            
            return 0

        return 1 + max_other_idx

    return max_idx
