import numpy as np
import librosa
import noisereduce as nr

TARGET_SAMPLE_RATE = 16000
TOP_DB = 30  # threshold (dB) below reference to trim as silence


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Prepares a raw audio chunk for the model.
    - resamples to TARGET_SAMPLE_RATE if needed
    - reduces background noise
    - trims leading/trailing silence
    - normalizes amplitude to [-1, 1]
    - ensures float32 dtype

    audio: 1D numpy array (mono)
    sample_rate: sample rate of `audio`
    returns: 1D numpy float32 array at TARGET_SAMPLE_RATE
    """
    audio = np.asarray(audio, dtype=np.float32)

    if sample_rate != TARGET_SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=TARGET_SAMPLE_RATE)

    audio = nr.reduce_noise(y=audio, sr=TARGET_SAMPLE_RATE, stationary=True)

    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak

    return audio.astype(np.float32)