import io
import torch
import torchaudio
import numpy as np

Target_sample_rate = 16000

def convert_to_standard_audio(file_bytes: bytes) -> np.ndarray:
    """
    Taking raw bytes of audio file from frontend
    Outputs: Clean 1D Numpy float32 array at exactly 16,000hz Mono
    """
    #decode audio bytes directly in memory
    audio_stream = io.BytesIO(file_bytes)
    waveform, original_sample_rate = torchaudio.load(audio_stream)
    if waveform[0].shape > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if original_sample_rate != Target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq = original_sample_rate,
            new_freq = Target_sample_rate
        )
        waveform = resampler(waveform)
    #Converting to 1D 32 bit floating point array
    audio_1d = waveform.squeeze(0).numpy.astype(np.float32)
    return audio_1d