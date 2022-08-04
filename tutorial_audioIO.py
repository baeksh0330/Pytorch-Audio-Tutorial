import torch
import torchaudio
import io
import os
import tarfile
import tempfile

import boto3
import matplotlib.pyplot as plt
import requests

from botocore import UNSIGNED
from botocore.config import Config
from IPython.display import Audio
from torchaudio.utils import download_asset

SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")

#metadata = torchaudio.info(SAMPLE_WAV)
#print(metadata)

# print(torch.__version__) # 1.12.0
# print(torchaudio.__version__) # 0.12.0
# print(torchaudio.get_audio_backend()) # PySoundFile 업데이트 후 soundfile 출력됨

# url = "https://download.pytorch.org/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav"
# with requests.get(url, stream=True) as response:
#     metadata = torchaudio.info(response.raw)
# print  # 런타임에러가 왜 나는데. . . . .

waveform, sample_rate = torchaudio.load(SAMPLE_WAV)

def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    # plt.savefig("test2.png") #이거 써주면 저장됨!
    plt.show(block=False)

plot_waveform(waveform, sample_rate)

def plot_specgram(waveform, sample_rate, title = "Spectrogram"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c + 1}")
    figure.suptitle(title)
    plt.show(block=False)

plot_specgram(waveform, sample_rate)
Audio(waveform.numpy()[0], rate = sample_rate)
plt.show()