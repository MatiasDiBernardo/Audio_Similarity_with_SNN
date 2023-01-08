import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def getRMS(x):

    return None


def getLoudness():
    
    return None


def getLoudness():
    return None

def eraAca():
    return None

def normVector(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def cosineSimilariy(x1, x2):
    x1 = normVector(x1)
    x2 = normVector(x2)

    return np.dot(x1, x2)

def euclideanDistance(x1, x2):
    dist = (x1 - x2)**2
    return np.sqrt(np.sum(dist)/len(x1) )

def nextPower2(N):
    return  2**(N-1).bit_length()

def DFT(x, w, N):
    """
	Analysis of a signal using the discrete Fourier transform
	x: input signal, w: analysis window, N: FFT size 
	returns mX, pX: magnitude and phase spectrum
	"""
    tol = 1e-14
    if w.size > N:  # raise error if window size bigger than fft size
        raise ValueError("Window size (M) is bigger than FFT size")

    hN = (N // 2) + 1  # size of positive spectrum, it includes sample 0
    hM1 = (w.size + 1) // 2  # half analysis window size by rounding
    hM2 = w.size // 2  # half analysis window size by floor
    fftbuffer = np.zeros(N)  # initialize buffer for FFT
    w = w / sum(w)  # normalize analysis window
    xw = x * w  # window the input sound
    fftbuffer[:hM1] = xw[hM2:]  # zero-phase window in fftbuffer
    fftbuffer[-hM2:] = xw[:hM2]
    X = fft(fftbuffer)  # compute FFT
    absX = abs(X[:hN])  # compute ansolute value of positive side
    absX[absX < np.finfo(float).eps] = np.finfo(float).eps  # if zeros add epsilon to handle log
    mXdB = 20 * np.log10(absX)  # magnitude spectrum of positive frequencies in dB
    return mXdB, absX

def loadAudio(pathAudio, fs=22050):
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
