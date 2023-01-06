import numpy as np
import librosa
import matplotlib.pyplot as plt

def getRMS(x):

    return None


def getLoudness():
    
    return None


def getLoudness():
    return None

def eraAca():
    return None

def loadAudio(pathAudio, fs):
    x, fs = librosa.load(pathAudio, sr=fs, mono=True)
    return x, fs

def cutSilenceAndNoise(x, fs, threshold):
    """
    x: audio signal, fs: samplerate,
    threshold: ratio to define noise or silence
    Rather than cut silence this function discard the low lever enery segments of the
    signal that will introuce numerical difference in the feature extractions, the value of
    threshold that I choose base on testing is 0.001
    """
    FL = 2048
    HS = 512
    rmsValueFrames = librosa.feature.rms(y=x, frame_length=FL, hop_length=HS)
    rmsValueFrames = rmsValueFrames[0,:]
    sizeProm = len(x)//len(rmsValueFrames)
    rmsExtended = np.zeros(len(x))
    for i in range(len(rmsValueFrames)):
        rmsExtended[i * sizeProm:i * sizeProm + sizeProm] = rmsValueFrames[i]
    rmsExtended[-sizeProm] = rmsValueFrames[-1]
    audioCut = x[rmsExtended > threshold]
    print("Long audio og:", len(x)/fs)
    print("Len del recortado: ", len(audioCut)/fs)
    return audioCut
