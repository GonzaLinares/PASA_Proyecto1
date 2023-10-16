import scipy.io as io
import matplotlib.pyplot as plt
import scipy.signal as sp
import IPython.display as ipd
import numpy as np
import scipy.linalg as lin
from numpy.fft import fft, rfft
from numpy.fft import fftshift, fftfreq, rfftfreq


def saveSignalAsWAV(name, signal, fs):

    io.wavfile.write(name, fs, signal)


def getRecordedSignals():

    path = ["recAAA1.wav", "recSHH1.wav"]

    fs1, aSignal = io.wavfile.read(path[0])
    fs2, shSignal = io.wavfile.read(path[1])

    aSignal = (aSignal.T)
    shSignal = (shSignal.T)

    fs = {"a": fs1, "sh": fs2}
    signal = {"a": aSignal, "sh": shSignal}

    return fs, signal, path


def play(signal, fs):
    audio = ipd.Audio(signal, rate=fs, autoplay=True)
    return audio


def plot_spectrogram(title, w, fs):
    ff, tt, Sxx = sp.spectrogram(w, fs=fs, nperseg=256, nfft=576)
    fig, ax = plt.subplots()
    ax.pcolormesh(tt, ff, Sxx, cmap='gray_r',
                  shading='gouraud')
    ax.set_title(title)
    ax.set_xlabel('t (sec)')
    ax.set_ylabel('Frequency (Hz)')
    ax.grid(True)


def getNextPowerOfTwo(len):
    return 2**(len*2).bit_length()


def get_optimal_params(x, M):

    lags = sp.correlation_lags(len(x), len(x))
    N = len(x)
    rx = sp.correlate(x, x)/N

    r = rx[np.array(M > lags) == np.array(lags >= 0)]

    p = rx[np.array(-1 >= lags) == np.array(lags >= -M)][-1::-1]

    wo = lin.solve_toeplitz(r, p)

    jo = np.var(x) - np.dot(p, wo)

    NMSE = jo/np.var(x)

    return wo, jo, NMSE


def periodogram_averaging(data, fs, L, padding_multiplier, window):
    wind = window(L)
    # Normalizamos la ventana para que sea asintoticamente libre de bias

    def getChuncks(lst, K): return [lst[i:i + K]
                                    for i in range(0, len(lst), K)][:-1]
    corrFact = np.sqrt(L/np.square(wind).sum())
    wind = wind*corrFact
    dataChunks = getChuncks(data, L)*wind
    fftwindowSize = L*padding_multiplier
    freqs = rfftfreq(fftwindowSize, 1/fs)
    periodogram = np.zeros(len(freqs))
    for i in range(len(dataChunks)):
        # Se van agregando al promediado los periodogramas de cada bloque calculado a partir de la FFT del señal en el tiempo
        periodogram = periodogram + \
            np.abs(rfft(dataChunks[i], fftwindowSize))**2/(L*len(dataChunks))

    return freqs, periodogram, len(dataChunks)


windows = ['boxcar', 'triang', 'parzen', 'bohman', 'blackman', 'nuttall',
           'blackmanharris', 'flattop', 'bartlett', 'barthann',
           'hamming', ('kaiser', 10), ('tukey', 0.25)]
