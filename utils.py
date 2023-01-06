import numpy as np
import librosa


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
    """
    print("si type es float norm el max siempre cerca de 1: ", np.max(x))
    rmsValueFrames = librosa.feature.rms(y=x, frame_length=2048, hop_length=512)
    
    return None