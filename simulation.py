import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.signal as sp
import scipy.io as io
from tqdm.notebook import tqdm
import scipy.io.wavfile as wav
import helper as hp
import pyroomacoustics as pra

def simulate():

    x = np.random.randn(999999) #Generamos ruido gaussiano
    y = np.random.randn(999999) #Mic de cancelacion
    fs = 48000

    #Seteamos los materiales de la habitacion
    m = pra.make_materials(ceiling=0.3,
                            floor=0.3,
                            east=0.8,
                            west=0.8,
                            north=0.8,
                            south=0.8,)

    #Elejimos las dimensiones de la habitacion en metros y creamos la habitacion
    #room_dim = [0.05, 0.05, 0.02]
    #room = pra.ShoeBox(room_dim, fs=fs, materials=m, air_absorption=True)

    height = 0.1
    roomCorners = np.array([[-1.0,-1.0, 0.4, 0.2, 0.2+np.sqrt(2)/20, 0.4+np.sqrt(2)/10, 0.6, 0.6], \
                            [ 0.1, 0.2, 0.2, 0.4, 0.4+np.sqrt(2)/20, 0.2, 0.2, 0.1]])
    room = pra.Room.from_corners(roomCorners, fs=fs)
    room.extrude(height)

    #Agregamos la fuente y el microfono
    micError = np.array([0.6, 0.15, height/2])
    room.add_microphone(micError)
                           
    room.add_source([-1.0, 0.15, height/2], signal=x)
    room.add_source([0.2+np.sqrt(2)/40, 0.4+np.sqrt(2)/40, height/2], signal=y)
    
    #Mostramos la habitacion
    fig, ax = room.plot(mic_marker_size=50)
    ax.set_xlim([-1.1, 0.6])
    ax.set_ylim([-1.1, 0.6])
    ax.set_zlim([0, 0.2])
    plt.show()

    # No se que hace esto
    room.image_source_model()

    #Computamos y mostramos la respuesta al impulso
    room.compute_rir()
    response = room.rir[0][0] #respuesta impulsiva
    plt.plot(response)
    
    return response

if __name__ == "__main__":

    simulate()
    simulate()