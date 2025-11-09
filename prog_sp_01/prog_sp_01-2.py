#prog_sp_01-2

#from google.colab import drive
#drive.mount('/content/drive')


import numpy as np    
from scipy import signal, fft
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display

print("prog 1-2b: speech spectrogram")

file_in = './prog_sp_01/aeiou.wav' # input speech wav file
x, fs = sf.read(file_in)  # x: input speech ([-1,1)), fs: sampling rate
print("data.shape:",x.shape)
print("data type:", type(x), type(x[0]))
print("sample rate:",fs)

f,  t,  Sxx  = signal.spectrogram(x, fs, nperseg=512, noverlap=384, \
               nfft=1024,window = signal.windows.hamming(512, sym=True))

f2, t2, Sxx2 = signal.spectrogram(x, fs, nperseg=64,  noverlap=48,  \
               nfft=1024,window = signal.windows.hamming(64, sym=True))

log_Sxx  = 10*np.log10(Sxx)
log_Sxx2 = 10*np.log10(Sxx2)

fig = plt.figure(figsize = (12,18), facecolor="lightgray")
plt.subplot(3,1,1)

time = np.arange(0,2.3,1/8000)       
plt.plot(time, x[0:18400])
plt.title("(1) Speech waveform (8kHz sampling)")
plt.xlim([0.,2.3])
plt.ylabel('amplitude')
plt.xlabel('time[s]')
plt.grid()

plt.subplot(3,1,2)
plt.pcolormesh(t, f, log_Sxx, cmap="jet",vmax=-50, vmin=-180) 
plt.title("(2) Speech spectrogram:high frequency resolution (nperseg=512,noverlap=384,nfft=1024)")
plt.xlim([0.,2.3])
plt.ylabel('frequency [Hz]')
plt.xlabel('time [s]')
#cbar = plt.colorbar(label="intensity [dB]") 

plt.subplot(3,1,3)
plt.pcolormesh(t2, f2, log_Sxx2, cmap="jet",vmax=-50, vmin=-180) 
plt.title("(3) Speech spectrogram:high time resolution (nperseg=64,noverlap=48,nfft=1024)")
plt.xlim([0.,2.3])
plt.ylabel('frequency [Hz]')
plt.xlabel('time [s]')
#cbar = plt.colorbar(label="intensity [dB]") 

plt.savefig("prog_sp_01-2.png")
plt.show()

#end
