import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy as sp
from ts06_pitch import ts06_pitch
from ts06_pitch_freq import ts06_pitch_freq
from ts06_median import ts06_median

#--- constants ---
Fs =  8000 # sampling frequency [Hz]
Lf =    240 # frame length [sample]
Ls =    40 # frame shift length [sample]
N_alf = 10 # LPC analysis order

#--- read input speech file ---
spfile = 'F01_mip_08k.wav'

try:
    # wavfile.read returns Fs and the data
    # Note: wavfile.read may return an array of integers. We convert to float 
    # and normalize if necessary.
    Fs_read, sp = wavfile.read(spfile)
    if Fs_read != Fs:
        print(f"Warning: File Fs ({Fs_read}) does not match constant Fs ({Fs}). Using file Fs.")
        Fs = Fs_read
        
    # Convert to float and normalize to -1.0 to 1.0 (typical for speech processing)
    if sp.dtype != np.float32 and sp.dtype != np.float64:
        sp = sp.astype(np.float32) / np.iinfo(sp.dtype).max

except FileNotFoundError:
    print(f"Error: Speech file '{spfile}' not found. Please ensure it is in the same directory.")
    exit()

# initialize variables
frame_pitch = []
frame_pow = []

# Calculation for number of frames (same as MATLAB's loop boundary)
# (length(sp)-(Lf-Ls))/Ls + 1 (for 1-based indexing)
# Python: 1 + floor((len(sp) - Lf) / Ls) 
num_frame = int(np.floor((len(sp) - Lf) / Ls)) + 1

for i in range(num_frame):
    
    fr_start = i * Ls
    fr_end   = i * Ls + Lf
    sp_fr = sp[fr_start : fr_end]
    
    # Ensure frame is exactly Lf samples long
    if len(sp_fr) < Lf:
        # This shouldn't happen with the num_frame calculation but is good practice
        break 
        
    # local synthesis pitch period (in samples)
    T0, Amax, corr_0 = ts06_pitch(sp_fr, Fs=Fs) 
    
    # MATLAB's fprintf output
    print(f"i={i+1} T0={T0:3d} Amax={Amax:7.2f} corr_0={corr_0:14.7f}")
    
    frame_pitch.append(T0)
    # frame_pow = sp_fr' * sp_fr (frame power)
    frame_pow.append(np.sum(sp_fr**2))   

frame_pitch = np.array(frame_pitch)
tp = np.arange(1, num_frame + 1) # Time points for frames (1-based index)

# --- Plotting ---
plt.figure(1, figsize=(10, 8))

# Subplot 1: Speech signal
plt.subplot(4, 1, 1)
plt.plot(sp)
plt.ylabel('speech')
plt.title(f"Pitch Analysis for {spfile}")

# Subplot 2: Pitch interval (T0 in samples)
plt.subplot(4, 1, 2)
plt.plot(tp, frame_pitch)
plt.ylabel('pitch interval (samples)')

# Subplot 3: Pitch frequency (1/T0 - not in Hz)
pitch_freq = ts06_pitch_freq(frame_pitch)

# Convert to Hz: F0 = Fs / T0. Since ts06_pitch_freq returns 1/T0, we multiply by Fs.
# We apply the Fs multiplier here for a meaningful plot (in Hz).
pitch_freq_hz = pitch_freq * Fs 

plt.subplot(4, 1, 3)
plt.plot(tp, pitch_freq_hz)
plt.ylabel('pitch freq. (Hz)')

# Subplot 4: Pitch frequency with median filter
# The original MATLAB code passes `pitch_freq` (1/T0) to the median filter.
# If we want to filter the Hz values, we should filter `pitch_freq_hz`.
# I will filter `pitch_freq_hz` for a more useful plot.
pitch_freq2_hz = ts06_median(pitch_freq_hz) 

plt.subplot(4, 1, 4)
plt.plot(tp, pitch_freq2_hz)
plt.ylabel('pitch freq. with median filt. (Hz)')
plt.xlabel('Frame Number')

plt.tight_layout()
plt.show()