import scipy.io as io
import matplotlib.pyplot as plt
import scipy.signal as sp
import IPython.display as ipd
from IPython.display import Audio, update_display
from ipywidgets import IntProgress
import pyroomacoustics as pra
import numpy as np
import scipy.linalg as lin
from numpy.fft import fft, rfft
from numpy.fft import fftshift, fftfreq, rfftfreq


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


def fxnlms_sim(w0, mu, P, S, S_hat, xgen, sound, orden_filtro, N=10000):
    w = w0
    J = np.zeros(N)
    e = np.zeros(N)
    d_hat = np.zeros(N)
    n = np.arange(0, N, 1, dtype=int)

    x = xgen(n)
    d = sp.lfilter(P[0], P[1], x)
    xf = sp.lfilter(S_hat[0], S_hat[1], x)

    xf = np.concatenate([np.zeros(orden_filtro-1), xf])
    zis = np.zeros(np.max([len(S[0]), len(S[1])])-1)
    ziw = np.zeros(orden_filtro-1)

    f = IntProgress(min=0, max=N)
    ipd.display(f)

    i = 0
    ipd.display(J[0], display_id='J')
    for n in range(N):

        y, ziw = sp.lfilter(w, [1], [x[n]], zi=ziw)

        y = y[0] + sound(n)

        d_hat_aux, zis = sp.lfilter(S[0], S[1], [y], zi=zis)
        d_hat[n] = d_hat_aux[0]
        e[n] = d[n] + d_hat_aux[0]
        J[n] = e[n] * e[n]
        w = w - mu * xf[n:n + orden_filtro][::-1] * e[n] / \
            (np.linalg.norm(xf[n:n + orden_filtro][::-1])**2 + 1e-6)
        if n//(N//100) > i:
            update_display(J[n], display_id='J')
            f.value = n
            i += 1

    return w, J, e, d, d_hat


def plot_results(results, P, S):
    w, J, e, d, d_hat = results
    plt.figure(figsize=(10, 25))

    plt.subplot(411)
    plt.title('SE vs n')
    plt.xlabel('n')
    plt.ylabel('Square error')
    plt.semilogy(J)
    J_smooth = sp.savgol_filter(J, 500, 3)

    plt.semilogy(J_smooth)

    plt.subplot(412)
    plt.title('Señal error vs n')
    plt.xlabel('n')
    plt.ylabel('error')
    plt.plot(e)

    plt.subplot(413)
    plt.title('Señal deseada negada y deseada estimada')
    plt.xlabel('n')
    plt.ylabel('Amplitud')
    plt.plot(-d, label='Señal deseada negada')
    plt.plot(d_hat, label='Señal deseada estimada')
    plt.legend()

    plt.subplot(414)

    freqs, s = sp.freqz(P[0], S[0], fs=48000, worN=10000)
    freqw, wps = sp.freqz(w, [1], fs=48000, worN=10000)
    plt.title('Comparativa entre modulo de W(z) y P(z)/S(z)')
    plt.plot(freqw, 20*np.log10(np.abs(wps)), label='W(z)')
    plt.plot(freqs, 20*np.log10(np.abs(s)), label='P(z)/S(z)')
    plt.legend()


def createRoom(print):

    fs = 48000

    x = np.random.randn(fs*5)  # Generamos ruido gaussiano
    y = np.random.randn(fs*5)  # Mic de cancelacion
    # Seteamos los materiales de la habitacion
    m = pra.Material('rough_concrete')

    height = 0.1
    roomCorners = np.array([[-1.0, -1.0, 0.4, 0.2, 0.2+np.sqrt(2)/20, 0.4+np.sqrt(2)/10, 0.6, 0.6],
                            [0.1, 0.2, 0.2, 0.4, 0.4+np.sqrt(2)/20, 0.2, 0.2, 0.1]])
    room = pra.Room.from_corners(roomCorners, fs=fs)
    room.materials = [m]*len(room.walls)
    room.extrude(height)

    # Agregamos la fuente y el microfono
    micError = np.array([0.6, 0.15, height/2])
    micRef = np.array([-0.6, 0.15, height/2])
    micArray = pra.beamforming.MicrophoneArray(
        R=np.array([micRef, micError]).T, fs=fs)
    room.add_microphone_array(micArray)

    room.add_source([-1.0, 0.15, height/2], signal=x)
    room.add_source([0.2+np.sqrt(2)/40, 0.4+np.sqrt(2)/40, height/2], signal=y)

    if (print):
        # Mostramos la habitacion
        fig, ax = room.plot(mic_marker_size=50)
        ax.set_xlim([-1.1, 0.6])
        ax.set_ylim([-1.1, 0.6])
        ax.set_zlim([0, 0.2])
        plt.show()

    return room


def getImpulse(room):

    room.image_source_model()

    # Computamos y mostramos la respuesta al impulso
    room.compute_rir()

    plt.figure()
    room.plot_rir()
    # plt.plot(room.rir[0][0])
    # plt.plot(room.rir[0][1])
    # plt.show()
    # plt.plot(room.rir[1][0]) #Del source
    # plt.plot(room.rir[1][1])
    print(type(np.array(room.rir)[0][0]))
    # respuesta impulsiva
    np.savetxt('RoomSim.txt', np.array(room.rir).ravel())
