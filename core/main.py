import numpy as np
import torch
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

# MODEL_DIR = "./models/wav2vec2-large-xlsr-53"  # path to the model directory
MODEL_DIR = "./models/wav2vec2-deepfake-voice-detector"  # path to the model directory
AI_INDEX = 1
HUMAN_INDEX = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_model = None
_feature_extractor = None


def _load():
    global _model, _feature_extractor
    if _model is None:
        _feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)
        _model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)
        _model.to(DEVICE)
        _model.eval()
        print(f"[core.main] Model loaded on: {DEVICE}")
    return _model, _feature_extractor


def predict(input_values: np.ndarray):
    """
    input_values: numpy float32 array, shape (1, sequence_length) or (sequence_length,)
    returns: (ai_probability, human_probability)
    """
    model, feature_extractor = _load()

    audio = np.asarray(input_values, dtype=np.float32).reshape(-1)

    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

    ai_probability = float(probs[AI_INDEX])
    human_probability = float(probs[HUMAN_INDEX])

    return ai_probability, human_probability