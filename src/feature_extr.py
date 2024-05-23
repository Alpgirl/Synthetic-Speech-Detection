import torch
from librosa import util
import librosa.feature
import numpy as np



# def logpowerspec():
#     pass

# def MFCC():
#     pass

# def CQCC():
#     pass


def get_MFCC(waveform, sample_rate):
    mfcc = librosa.feature.mfcc(y=waveform.numpy().squeeze(), sr=sample_rate, n_mfcc=20)
    return torch.from_numpy(mfcc)

def get_CQCC(waveform, sample_rate):
    cqcc = librosa.feature.chroma_cqt(y=waveform.numpy().squeeze(), sr=sample_rate)
    return torch.from_numpy(cqcc)

def get_LPS(waveform, sample_rate):
    stft = librosa.core.stft(waveform.numpy().squeeze())
    lps = librosa.amplitude_to_db(np.abs(stft))
    return torch.from_numpy(lps)
