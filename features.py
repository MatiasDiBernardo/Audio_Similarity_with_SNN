import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.special import softmax
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import ShortTermFeatures
#First category of descriptors, less computantion and general descriptores
def featuresShort(x, fs):
    """
    x: audio signal mono
    fs: sample rate

    This function returns the most siginificant and easy to evaluate features to achieve
    a fast first comparison only on envelope characteristics

    """
    timeSignal = len(x)/fs
    rmsSignal = librosa.feature.rms(x) 
    rmsSignal = np.mean(rmsSignal[0,:])
    #Here use the function from sms
    fftSignal = fft(x)

    N = len(fftSignal) 
    fc1 = int(500*N/fs) + 1
    fc2 = int(2500*N/fs) + 1

    fftSignalLow = fftSignal[:fc1]
    fftSignalMid = fftSignal[fc1:fc2]
    fftSignalHigh = fftSignal[fc2:]

    features = [timeSignal, rmsSignal, fftSignalLow, fftSignalMid, fftSignalHigh]
    print(features)
    print(softmax(features))
    
    return features 
#Second category, all the different and most representative type of descriptors
#It uses the pyAudioAnalysis feature extractor

def featuresAll(x, Fs):
    #Test if the read wav from librosa or from this are the same
    #[Fs, x] = audioBasicIO.read_audio_file("sample.wav")
    F, f_names = ShortTermFeatures.feature_extraction(x, Fs, 0.050*Fs, 0.025*Fs)
    plt.subplot(2,1,1); plt.plot(F[0,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[0]) 
    plt.subplot(2,1,2); plt.plot(F[1,:]); plt.xlabel('Frame no'); plt.ylabel(f_names[1]); plt.show()
    
    return None

#Third category features obtenid by the neuronal network, this process is inside the SNN model 
#The final archeyectura takes two audios as the input and return a value of similarity