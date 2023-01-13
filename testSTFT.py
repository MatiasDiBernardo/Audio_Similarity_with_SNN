import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def fixLength(x, fs):
    smpls3sec = fs * 3
    if len(x) <= smpls3sec:
        zeroPad = np.zeros(smpls3sec - len(x))
        return np.concatenate([x, zeroPad])
    else:
        return x[:smpls3sec]

def stft(x, fs):
    stft = librosa.stft(x, n_fft=1024, hop_length=512,
    win_length=1001, window="blackman")
    stft = np.abs(stft)
    #stft = librosa.amplitude_to_db(stft, ref=np.max)
    #stft = np.transpose(stft)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
    y_axis='log', x_axis='time', sr=fs, ax=ax)
    print("Shape de la stft: ", stft.shape) 
    #plt.imshow(stft)
    #plt.show()
    ax.set_axis_off()
    fig.colorbar(img)
    fig.delaxes(fig.axes[1])
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    N = data.shape[0]
    M = data.shape[1]
    for i in range(N):
        allwhite = M * 255
        if sum(data[i,:,0]) != allwhite:
            upVertCut = i
            break

    for i in range(N):
        allwhite = M * 255
        if sum(data[N - 1 - i,:,0]) != allwhite:
            lowVertCut =  N - i - 1
            break

    for i in range(M):
        allwhite = N * 255
        if sum(data[:,i,0]) != allwhite:
            rightHorCut = i
            break
    for i in range(M):
        allwhite = N * 255
        if sum(data[:,M - i - 1,0]) != allwhite:
            leftHorCut = M - i - 1
            break
    

    print("Shape img", data.shape)
    data = data[upVertCut:lowVertCut, rightHorCut:leftHorCut, :]
    print("Shape img 2", data.shape)
    plt.show()
    return data

x, fs = librosa.load("Test\kick1.wav", sr=44100, mono=True)
x = fixLength(x, fs)

data = stft(x[:fs], fs)
plt.figure("dos")
plt.imshow(data)
plt.show()