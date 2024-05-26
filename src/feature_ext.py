import torch
from librosa import util
import librosa.feature
import numpy as np


def get_MFCC(waveform, sample_rate):
    waveform = waveform.cpu().numpy()  # Move tensor to CPU and convert to numpy
    batch_size = waveform.shape[0]
    features = []
    for i in range(batch_size):
        mfcc = librosa.feature.mfcc(y=waveform[i].squeeze(), sr=sample_rate, n_mfcc=20)
        features.append(torch.from_numpy(mfcc).unsqueeze(0))  # Shape (1, n_mfcc, time)
    return torch.cat(features, dim=0)  # Shape (batch_size, n_mfcc, time)

def get_CQCC(waveform, sample_rate):
    waveform = waveform.cpu().numpy()  # Move tensor to CPU and convert to numpy
    batch_size = waveform.shape[0]
    features = []
    for i in range(batch_size):
        cqcc = librosa.feature.chroma_cqt(y=waveform[i].squeeze(), sr=sample_rate)
        features.append(torch.from_numpy(cqcc).unsqueeze(0))  # Shape (1, n_cqcc, time)
    return torch.cat(features, dim=0)  # Shape (batch_size, n_cqcc, time)

def get_LPS(waveform, sample_rate):
    waveform = waveform.cpu().numpy()  # Move tensor to CPU and convert to numpy
    batch_size = waveform.shape[0]
    features = []
    for i in range(batch_size):
        stft = librosa.core.stft(waveform[i].squeeze())
        lps = librosa.amplitude_to_db(np.abs(stft))
        features.append(torch.from_numpy(lps).unsqueeze(0))  # Shape (1, n_lps, time)
    return torch.cat(features, dim=0)  # Shape (batch_size, n_lps, time)


