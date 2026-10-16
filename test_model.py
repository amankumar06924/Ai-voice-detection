import os
import csv
import time

import torch
import librosa

from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "./models/wav2vec2-deepfake-voice-detector"

TEST_DATA_PATH = "./Audio_data/for-original/for-original/testing"

RESULTS_FILE = "results.csv"

SAMPLE_RATE = 16000

THRESHOLD = 0.5


# ============================================================
# DEVICE
# ============================================================

if torch.cuda.is_available():

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


print("=" * 70)
print("             AI VOICE DEEPFAKE MODEL EVALUATION")
print("=" * 70)

print("\nDevice:", device)

if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )

    print(
        "CUDA:",
        torch.version.cuda
    )


# ============================================================
# CHECK PATHS
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\nERROR: Model not found!")

    print(
        "Expected:",
        os.path.abspath(MODEL_PATH)
    )

    exit()


if not os.path.exists(TEST_DATA_PATH):

    print("\nERROR: test_data folder not found!")

    print(
        "Expected:",
        os.path.abspath(TEST_DATA_PATH)
    )

    exit()


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

load_start = time.time()


feature_extractor = AutoFeatureExtractor.from_pretrained(
    MODEL_PATH
)


model = AutoModelForAudioClassification.from_pretrained(
    MODEL_PATH
)


model = model.to(device)

model.eval()


load_time = time.time() - load_start


print(
    f"Model loaded in {load_time:.2f} seconds"
)


# ============================================================
# DISPLAY LABELS
# ============================================================

print("\nModel labels:")

for label_id, label_name in model.config.id2label.items():

    print(
        f"  {label_id} -> {label_name}"
    )


# ============================================================
# FIND AUDIO FILES
# ============================================================

audio_extensions = (
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a"
)


files = []


# ------------------------------------------------------------
# AI folder
# ------------------------------------------------------------

ai_folder = os.path.join(
    TEST_DATA_PATH,
    "ai"
)


if os.path.exists(ai_folder):

    for filename in os.listdir(ai_folder):

        if filename.lower().endswith(audio_extensions):

            files.append(
                (
                    os.path.join(ai_folder, filename),
                    1,
                    "AI"
                )
            )


# ------------------------------------------------------------
# Human folder
# ------------------------------------------------------------

human_folder = os.path.join(
    TEST_DATA_PATH,
    "human"
)


if os.path.exists(human_folder):

    for filename in os.listdir(human_folder):

        if filename.lower().endswith(audio_extensions):

            files.append(
                (
                    os.path.join(human_folder, filename),
                    0,
                    "HUMAN"
                )
            )


# ============================================================
# CHECK FILES
# ============================================================

if len(files) == 0:

    print("\nERROR: No audio files found.")

    print("\nExpected structure:")

    print(
        """
test_data/
├── ai/
│   ├── audio1.wav
│   └── audio2.wav
│
└── human/
    ├── audio1.wav
    └── audio2.wav
"""
    )

    exit()


print(
    f"\nFound {len(files)} audio files."
)

print(
    f"AI files    : {sum(1 for x in files if x[2] == 'AI')}"
)

print(
    f"Human files : {sum(1 for x in files if x[2] == 'HUMAN')}"
)


# ============================================================
# STORAGE
# ============================================================

y_true = []

y_pred = []

results = []


# ============================================================
# RUN TEST
# ============================================================

print("\n")
print("=" * 70)
print("                         TESTING")
print("=" * 70)


total_start = time.time()


for index, (file_path, actual_label, actual_name) in enumerate(files, 1):

    filename = os.path.basename(file_path)


    print(
        f"\n[{index}/{len(files)}] {filename}"
    )

    try:

        # ----------------------------------------------------
        # Load audio
        # ----------------------------------------------------

        audio, sr = librosa.load(
            file_path,
            sr=SAMPLE_RATE,
            mono=True
        )


        duration = len(audio) / SAMPLE_RATE


        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        inputs = feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )


        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }


        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        if device.type == "cuda":

            torch.cuda.synchronize()


        inference_start = time.time()


        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.nn.functional.softmax(
                outputs.logits,
                dim=-1
            )


        if device.type == "cuda":

            torch.cuda.synchronize()


        inference_time = (
            time.time() - inference_start
        )


        # ----------------------------------------------------
        # Probabilities
        # ----------------------------------------------------

        prob_real = probabilities[0][0].item()

        prob_fake = probabilities[0][1].item()


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        if prob_fake >= THRESHOLD:

            predicted_label = 1

            predicted_name = "AI"

        else:

            predicted_label = 0

            predicted_name = "HUMAN"


        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        y_true.append(actual_label)

        y_pred.append(predicted_label)


        correct = (
            actual_label == predicted_label
        )


        results.append(
            {
                "file": filename,
                "actual": actual_name,
                "predicted": predicted_name,
                "real_probability": prob_real,
                "ai_probability": prob_fake,
                "confidence": max(
                    prob_real,
                    prob_fake
                ),
                "duration": duration,
                "inference_time_ms":
                    inference_time * 1000,
                "correct": correct
            }
        )


        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        status = "✓ CORRECT" if correct else "✗ WRONG"


        print(
            f"Actual     : {actual_name}"
        )

        print(
            f"Prediction : {predicted_name}"
        )

        print(
            f"Human      : {prob_real * 100:.2f}%"
        )

        print(
            f"AI         : {prob_fake * 100:.2f}%"
        )

        print(
            f"Confidence : "
            f"{max(prob_real, prob_fake) * 100:.2f}%"
        )

        print(
            f"Duration   : {duration:.2f}s"
        )

        print(
            f"Time       : "
            f"{inference_time * 1000:.2f} ms"
        )

        print(
            f"Result     : {status}"
        )


    except Exception as e:

        print(
            f"ERROR processing {filename}: {e}"
        )


# ============================================================
# TOTAL TIME
# ============================================================

total_time = (
    time.time() - total_start
)


# ============================================================
# METRICS
# ============================================================

print("\n")
print("=" * 70)
print("                         RESULTS")
print("=" * 70)


accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)


recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)


f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print(
    f"\nAccuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)


print("\nConfusion Matrix")

print(
    """
                 Predicted
                 HUMAN   AI
Actual HUMAN      TN     FP
       AI         FN     TP
"""
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "HUMAN",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    RESULTS_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as csvfile:

    fieldnames = [
        "file",
        "actual",
        "predicted",
        "real_probability",
        "ai_probability",
        "confidence",
        "duration",
        "inference_time_ms",
        "correct"
    ]


    writer = csv.DictWriter(
        csvfile,
        fieldnames=fieldnames
    )


    writer.writeheader()


    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

correct_count = sum(
    1 for result in results
    if result["correct"]
)


wrong_count = len(results) - correct_count


avg_inference_time = sum(
    result["inference_time_ms"]
    for result in results
) / len(results)


print("\n")
print("=" * 70)
print("                         SUMMARY")
print("=" * 70)


print(
    f"\nTotal files      : {len(results)}"
)

print(
    f"Correct          : {correct_count}"
)

print(
    f"Incorrect        : {wrong_count}"
)

print(
    f"Accuracy         : {accuracy * 100:.2f}%"
)

print(
    f"Average inference: {avg_inference_time:.2f} ms"
)

print(
    f"Total test time  : {total_time:.2f} seconds"
)

print(
    f"\nDetailed results saved to:"
)

print(
    os.path.abspath(RESULTS_FILE)
)


print("\nDone!")