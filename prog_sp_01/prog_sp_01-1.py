#prog 1-1r

#from google.colab import drive
#drive.mount('/content/drive')


import numpy as np    
from scipy import signal, fft
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display

print("prog 1-1r: speech spectrum")

file_in = './prog_sp_01/aiueo3_08k.wav' # input speech wav file
x, fs = sf.read(file_in)  # x: input speech ([-1,1)), fs: sampling rate
print("data.shape:",x.shape)
print("data type:", type(x), type(x[0]))
print("samplerate:",fs)

len = 512
s = x[8000:8000+len] #cut out speech with len
win_ham = signal.windows.hamming(len,sym=True)
sw = s * win_ham
y = fft.fft(sw)
y_abs = np.abs(y)

win_rect = np.ones(len)
sw_rect = s*win_rect
y_rect = fft.fft(sw_rect)
y_rect_abs = np.abs(y_rect)
#plt.plot(win_rect)
#plt.plot(20*np.log10(y_rect_abs[0:len//2]))

#plot signals

fig = plt.figure(figsize = (20,12), facecolor="lightgray")

ax1 = fig.add_subplot(2,3,1)
ax1.plot(s,color="b",linewidth=1)
ax1.set_title("(1) original signal")
ax1.set_xlabel("time[sample]")
ax1.set_ylabel("amplitude")
ax1.grid()

ax2 = fig.add_subplot(2,3,2)
ax2.plot(win_ham,color="orange",linewidth=1)
ax2.set_title("(2) window")
ax2.set_xlabel("time[sample]")
ax2.set_ylabel("amplitude")
ax2.grid()

ax3 = fig.add_subplot(2,3,3)
ax3.plot(sw,color="b",linewidth=1)
ax3.set_title("(3) windowed signal")
ax3.set_xlabel("time[sample]")
ax3.set_ylabel("amplitude")
ax3.grid()

ax4 = fig.add_subplot(2,3,4)
ax4.plot(20*np.log10(y_abs),color="green",linewidth=1)
ax4.set_title("(4)fourier transfom")
ax4.set_xlabel("frequency[sample]")
ax4.set_ylabel("amplitude[dB]")
ax4.grid()

ax5 = fig.add_subplot(2,3,5)
ax5.plot(20*np.log10(y_abs[0:len//2]),color="green",linewidth=1)
ax5.set_title("(5)fourier transfom(half range)")
ax5.set_xlabel("frequency[sample]")
ax5.set_ylabel("amplitude[dB]")
ax5.grid()

ax6 = fig.add_subplot(2,3,6)
ax6.plot(20*np.log10(y_abs[0:len//2]),color="green",linewidth=1)
ax6.plot(20*np.log10(y_rect_abs[0:len//2]),color="red",linewidth=1)
ax6.set_title("(6)FT of win and non-win signals")
ax6.set_xlabel("frequency[sample]")
ax6.set_ylabel("amplitude[dB]")
ax6.grid()

plt.savefig("prog_sp_01-1.png")
plt.show()

#end
