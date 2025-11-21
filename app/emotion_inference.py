import torch
from model_loader import transform, calibrate_probs, decision_rule, IDX_TO_LABEL, device

def infer_emotion(model, frame_bgr) -> str:
    frame_rgb = frame_bgr[:, :, ::-1]

    tensor = transform(frame_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    adj = calibrate_probs(probs)
    final_idx = decision_rule(adj)
    emotion_label = IDX_TO_LABEL[final_idx]

    # logs locales
    print("Probabilidades ajustadas:", [f"{p.item():.3f}" for p in adj])
    print("Emoción detectada:", emotion_label)

    return emotion_label
