import os
import torch  # noqa: F401  (import before onnxruntime to preload CUDA/cuDNN DLLs)
import numpy as np
import onnxruntime as ort
import librosa

MODEL_PATH = "./models/voice_detector.onnx"
TEST_DIR = "./test_data/testing"
LABELS = {0: "human", 1: "ai"}  # adjust order if your model differs

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
print(f"Using provider: {session.get_providers()[0]}\n")

correct = 0
total = 0

for true_label in ["ai", "human"]:
    folder = os.path.join(TEST_DIR, true_label)

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(folder, filename)

        audio, sr = librosa.load(filepath, sr=16000)
        input_values = audio.reshape(1, -1).astype(np.float32)

        logits = session.run(None, {"input_values": input_values})[0]
        pred_id = int(np.argmax(logits, axis=-1)[0])
        probs = np.exp(logits) / np.exp(logits).sum()

        pred_label = LABELS.get(pred_id, pred_id)
        is_correct = pred_label == true_label
        correct += is_correct
        total += 1

        print(f"{filepath:30s} true={true_label:6s} pred={pred_label:6s} "
              f"conf={probs[0][pred_id]:.4f} {'OK' if is_correct else 'WRONG'}")

print(f"\nAccuracy: {correct}/{total} ({100 * correct / total:.1f}%)")