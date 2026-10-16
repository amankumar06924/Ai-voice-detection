import os
import torch

from transformers import AutoModelForAudioClassification


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./models/wav2vec2-deepfake-voice-detector"

ONNX_PATH = "./models/voice_detector.onnx"


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("       WAV2VEC2 → ONNX CONVERSION")
print("=" * 60)

print("\nLoading model...")

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_PATH
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

print("\nModel information:")

print("Model type:", model.config.model_type)

print("Number of labels:", model.config.num_labels)

print("Labels:")

for label_id, label_name in model.config.id2label.items():

    print(
        f"  {label_id} -> {label_name}"
    )


# ============================================================
# DUMMY INPUT
# ============================================================

# 5 seconds of audio at 16 kHz
#
# 5 × 16000 = 80000 samples

dummy_audio = torch.randn(
    1,
    80000,
    dtype=torch.float32
)


# Attention mask
#
# 1 = valid audio sample

dummy_attention_mask = torch.ones(
    1,
    80000,
    dtype=torch.long
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(ONNX_PATH),
    exist_ok=True
)


# ============================================================
# ONNX EXPORT
# ============================================================

print("\nConverting model to ONNX...")

print("Input shape:")
print("  audio:", tuple(dummy_audio.shape))

print("\nOutput:")
print("  logits")


with torch.no_grad():

    torch.onnx.export(
        model,
        (
            dummy_audio,
            dummy_attention_mask
        ),
        ONNX_PATH,

        input_names=[
            "input_values",
            "attention_mask"
        ],

        output_names=[
            "logits"
        ],

        dynamic_axes={

            "input_values": {
                0: "batch_size",
                1: "audio_length"
            },

            "attention_mask": {
                0: "batch_size",
                1: "audio_length"
            },

            "logits": {
                0: "batch_size"
            }
        },

        opset_version=17,

        do_constant_folding=True
    )


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.exists(ONNX_PATH):

    raise RuntimeError(
        "ONNX conversion failed."
    )


file_size_mb = (
    os.path.getsize(ONNX_PATH)
    / (1024 * 1024)
)


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 60)

print("ONNX conversion completed!")

print("=" * 60)

print("\nONNX model:")

print(
    os.path.abspath(ONNX_PATH)
)

print(
    f"\nFile size: {file_size_mb:.2f} MB"
)

print("\nNext file:")
print("models/voice_detector.onnx")