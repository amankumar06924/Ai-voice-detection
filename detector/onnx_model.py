import os
import torch

from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_DIR = "./models/wav2vec2-deepfake-voice-detector"
ONNX_OUTPUT_PATH = "./models/voice_detector.onnx"

# Wav2Vec2 expects raw waveform at 16kHz. Dummy length only affects
# the trace; dynamic_axes below lets real inputs be any length.
DUMMY_AUDIO_LENGTH = 16000  # 1 second @ 16kHz
OPSET_VERSION = 14


# ============================================================
# CHECK MODEL FOLDER
# ============================================================

if not os.path.isdir(MODEL_DIR):

    raise FileNotFoundError(
        f"Model folder not found: {MODEL_DIR}"
    )


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================

print("=" * 60)
print("           PYTORCH -> ONNX CONVERSION")
print("=" * 60)

print("\nLoading PyTorch model and feature extractor...")

model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR)
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_DIR)

model.eval()

print("Model loaded successfully.")
print(f"Model class : {model.__class__.__name__}")
print(f"Num labels  : {model.config.num_labels}")


# ============================================================
# BUILD DUMMY INPUT
# ============================================================

print("\nBuilding dummy input for tracing...")

dummy_audio = torch.randn(1, DUMMY_AUDIO_LENGTH)

# Feature extractor normalizes the raw waveform the same way it
# would at inference time (zero mean / unit variance).
inputs = feature_extractor(
    dummy_audio.numpy()[0],
    sampling_rate=16000,
    return_tensors="pt",
)

input_values = inputs["input_values"]

print(f"Dummy input shape: {tuple(input_values.shape)}")


# ============================================================
# EXPORT TO ONNX
# ============================================================

print("\nExporting model to ONNX...")

os.makedirs(os.path.dirname(ONNX_OUTPUT_PATH), exist_ok=True)

torch.onnx.export(
    model,
    (input_values,),
    ONNX_OUTPUT_PATH,
    export_params=True,
    opset_version=OPSET_VERSION,
    do_constant_folding=True,
    input_names=["input_values"],
    output_names=["logits"],
    dynamic_axes={
        "input_values": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"},
    },
)

print(f"ONNX model saved to: {ONNX_OUTPUT_PATH}")


# ============================================================
# SANITY CHECK WITH ONNX RUNTIME
# ============================================================

print("\nVerifying exported model with ONNX Runtime...")

import onnxruntime as ort

session = ort.InferenceSession(
    ONNX_OUTPUT_PATH,
    providers=["CPUExecutionProvider"],
)

ort_inputs = {"input_values": input_values.numpy()}
ort_outputs = session.run(None, ort_inputs)

with torch.no_grad():
    torch_outputs = model(input_values).logits

print(f"PyTorch output shape : {tuple(torch_outputs.shape)}")
print(f"ONNX output shape    : {ort_outputs[0].shape}")

max_diff = (torch_outputs.numpy() - ort_outputs[0]).__abs__().max()
print(f"Max abs difference   : {max_diff:.6f}")

if max_diff < 1e-3:
    print("Outputs match within tolerance. Conversion successful.")
else:
    print("WARNING: outputs differ more than expected. Review the export.")

print("\nConversion complete.")