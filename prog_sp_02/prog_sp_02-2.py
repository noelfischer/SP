#prog_sp_02-2.py
#wav file read and plot

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import IPython.display

#--- main

file = './aiueo2_16k_work.wav'

x_sig, fs = sf.read(file)

print("fs=", fs)
len_x_sig = len(x_sig)         # x.size
print("len_x_sig=", len_x_sig)

# exstract a frame of vowel /a/

t_start = 1.0 #[s] start time of a frame
t_end   = 1.2 #[s] start time of a frame  0.2sec -> 0.2 * 16kHz = 3.2k samples
x_frame = x_sig[int(t_start*fs):int(t_end*fs)]

mean_x_frame = np.mean(x_frame)
print("mean_x_frame=", mean_x_frame)

t = np.arange(int(t_start*fs),int(t_end*fs)) / fs
print(t.size)
plt.plot(t, x_frame)
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.title("vowel /a/ (1 frame)")
plt.grid()
#plt.savefig("./osnei3-1_vowel_wave.png")
plt.show()

t_center = (t_start + t_end) / 2
print(t_center)
x_center = x_sig[int(t_start + t_center*fs-500) : int(t_start + t_center*fs+500)]
t = np.arange(int(t_start + t_center*fs-500), int(t_start + t_center*fs+500)) / fs
print(t.size)
print(len(x_center))

plt.plot(t, x_center)
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.title("vowel /a/ (center)")
plt.grid()
#plt.savefig("./osnei3-1_vowel_wave2.png")
plt.show()

x_corr_org = np.correlate(x_frame, x_frame, mode='full')
x_corr = x_corr_org / np.max(x_corr_org)       #normalized corr.
x_corr_max = np.max(x_corr)
pos_x_corr_0 = np.argmax(x_corr)
print(x_corr.size)
print(x_corr_max, pos_x_corr_0)

t = np.arange(-500, 500) / fs
plt.plot(t, x_corr[pos_x_corr_0-500 : pos_x_corr_0+500])
plt.xlabel("time [s]")
plt.ylabel("amplitude")
plt.title("normalized auto-correlation")
plt.grid()
#plt.savefig("./osnei3-1_vowel_corr.png")
plt.show()

x_corr_tmp  = x_corr
x_corr_tmp[0:pos_x_corr_0 + 50] = 0
x_corr_max2 = np.max(x_corr_tmp)
pos_x_corr_1 = np.argmax(x_corr_tmp)
pitch_interval = (pos_x_corr_1 - pos_x_corr_0) / fs #[s]
print(x_corr_max2, pos_x_corr_1, pitch_interval)
print("pitch interval [ms]=",pitch_interval*1000)
print("pitch frequency [Hz]=",1/pitch_interval)

#eof