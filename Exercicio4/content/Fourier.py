#%% SOM
from matplotlib import pyplot as plt
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

filename = '..\\Imagens\\StarWars60.wav'
# Extract data and sampling rate from file
data60, fs = sf.read(filename, dtype='float32')  
sd.play(data60, fs)
status = sd.wait()  # Wait until file is done playing
data10 = # complete aqui#
#sd.play(data10, fs)
#status = sd.wait()  # Wait until file is done playing
#write('output.wav', fs, myrecording)  # Save as WAV file 
write('..\\Imagens\\StarWars10.wav', fs, data10)

plt.plot(data10)
datahat = np.fft.fft(data10)
plt.plot(abs(datahat), 'b')

#%% IMAGEM
import cv2 as cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('..\\Imagens\\frutas.jpg')
cv2.imshow('Frutas', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# %%
impb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imghat = np.fft.fft2(impb)
aux = np.abs(imghat)
aux = aux/aux.max()
cv2.imshow('fourier', aux*255)
cv2.waitKey(0)
cv2.destroyAllWindows()