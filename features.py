import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.special import softmax
from scipy.signal import get_window
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
import utils as UF
#First category of descriptors, less computantion and general descriptores
def featuresShort(x, fs):
    """
    x: audio signal mono
    fs: sample rate

    This function returns the most siginificant and easy to evaluate features to achieve
    a fast first comparison only on envelope characteristics, the analysis is on the general
    aspects of the sound so no windowing is applied

    """
    timeSignal = len(x)/fs
    rmsSignal = librosa.feature.rms(x) 
    rmsSignal = np.mean(rmsSignal[0,:])

    xSpace = len(x) // 8
    xCut = x[xSpace:-xSpace]
    w = get_window("blackman", len(xCut)) 
    NFFT = UF.nextPower2(len(xCut)) 
    fftSignaldB, fftSignal = UF.DFT(xCut, w, NFFT)

    N = len(fftSignal) 
    fc1 = int(500*N/fs) + 1
    fc2 = int(2500*N/fs) + 1

    fftSignalLow = np.mean(fftSignal[:fc1])
    fftSignalMid = np.mean(fftSignal[fc1:fc2])
    fftSignalHigh = np.mean(fftSignal[fc2:])

    features = [timeSignal, rmsSignal, fftSignalLow, fftSignalMid, fftSignalHigh]
    
    return np.array(features) 
#Second category, all the different and most representative type of descriptors
#It uses the pyAudioAnalysis feature extractor

def featuresAll(x, Fs):
    #Test if the read wav from librosa or from this are the same
    #[Fs, x] = audioBasicIO.read_audio_file("sample.wav")
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    print("This are the features: ", F)
    print("This are the names: ", f_names)
    plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]) 
    plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()
    
    return None
    
#Third category features obtenid by the neuronal network, this process is inside the SNN model 
def SNNModel(x):
    return None
    
#The final archeyectura takes two audios as the input and return a value of similarity
def similarityAudios(x1, fs1,  x2, fs2, type):
    """
    x1: Audio to compare
    x2: Audio to compare
    type: Type of method to compare where 1 is short features, 2 is all features
    and 3 is the neuronal model

    Return: float value with the degree of similarity 
    """
    if type == 1:
        features1 = featuresShort(x1, fs1)
        features2 = featuresShort(x2, fs2)

    similarityDot = UF.cosineSimilariy(features1, features2)
    similarityEuc = UF.euclideanDistance(features1, features2)

    print("Norm 1: ", features1)
    print("Norm 2: ", features2)
    
    return similarityDot, similarityEuc 
    
pathAudio1 = "Test\sounds_flute-A4.wav"
pathAudio2 = "Test\sounds_trumpet-A4.wav"
audio1, fs1 = UF.loadAudio(pathAudio1)
audio2, fs2 = UF.loadAudio(pathAudio2)

rta, rta2 = similarityAudios(audio1, fs1, audio2, fs2, type=1)
print("Similarity value cos: ", rta)
print("Similarity value euc: ", rta2)
