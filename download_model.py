from huggingface_hub import snapshot_download

# MODEL_ID = "facebook/wav2vec2-large-xlsr-53"
MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
# MODEL_ID = "zennnsenInpact/wav2vec2-deepfake-voice-detector"

# MODEL_PATH = "./models/wav2vec2-large-xlsr-53"
MODEL_PATH = "./models/wav2vec2-deepfake-voice-detector"

print("Downloading model...")
print(f"Model: {MODEL_ID}")
print(f"Saving to: {MODEL_PATH}")
print()

path = snapshot_download(
    repo_id=MODEL_ID,
    local_dir=MODEL_PATH
)

print("\nDownload completed!")
print(f"Model saved at: {path}")