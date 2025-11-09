import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal.windows import hamming
from prog_sp_04 import lpc_autocorr
from ts06_pitch import ts06_pitch
from ts06_pitch_freq import ts06_pitch_freq
from ts06_median import ts06_median

#--- constants ---
Fs =  8000 # sampling frequency [Hz]
Lf =    240 # frame length [sample]
Ls =    40 # frame shift length [sample]
N_alf = 10 # LPC analysis order

dft_size = 1024
A = np.zeros(dft_size // 2 + 1)
frequency = np.linspace(0, Fs / 2, dft_size // 2 + 1)

#--- read input speech file ---
spfile = 'aiueo3_08k.wav'

try:
    Fs_read, sp = wavfile.read(spfile)
    if Fs_read != Fs:
        print(f"Warning: File Fs ({Fs_read}) does not match constant Fs ({Fs}). Using file Fs.")
        Fs = Fs_read
        
    if sp.dtype != np.float32 and sp.dtype != np.float64:
        sp = sp.astype(np.float32) / np.iinfo(sp.dtype).max

except FileNotFoundError:
    print(f"Error: Speech file '{spfile}' not found. Please ensure it is in the same directory.")
    exit()

# initialize variables
frame_pitch = []
frame_pow = []

# Calculation for number of frames
num_frame = int(np.floor((len(sp) - Lf) / Ls)) + 1

# Setup figure 3 for LPC spectra
plt.figure(3, figsize=(10, 10))

for i in range(num_frame):
    
    fr_start = i * Ls
    fr_end   = i * Ls + Lf
    sp_fr = sp[fr_start : fr_end]
    
    if len(sp_fr) < Lf:
        break
        
    # LPC analysis with Hamming window
    # ai = [1, a1, ..., ap]
    ai, sigma_square = lpc_autocorr(sp_fr * hamming(Lf), N_alf)
    sigma = np.sqrt(sigma_square)
    
    # 1. Calculate LPC Spectrum
    # X = fft(alf, dft_size)
    X = np.fft.fft(ai, dft_size)
    
    # Take the magnitude of the first half (up to Nyquist)
    X_half = X[:dft_size // 2 + 1]
    
    # A(k) = 20 * log10(|X(k)|)
    # The LPC *filter* transfer function is 1/A(z). The spectrum of A(z) is |A(jw)|.
    # The spectrum of the speech *signal* is proportional to 1/|A(jw)|.
    # Speech Spectrum (dB) = -20 * log10(|A(jw)|) + Constant
    # The MATLAB script uses: A(k)=20*log10(abs(X(k))), which plots |A(jw)| in dB.
    # The *inverted* spectrum is usually plotted for the speech signal.
    # To match the MATLAB code's literal calculation:
    A = 20 * np.log10(np.abs(X_half))
    
    # MATLAB's plot offset/scaling logic (highly customized for visualization):
    offset = 0.1 * A[0]
    A_scaled = 0.1 * A + offset + (i + 1) # i+1 is the 1-based frame index
    
    # Plotting the LPC spectra for the five vowels (using 1-based index i+1)
    # Vowel /a/: frames 201 to 249
    if (i + 1 > 200) and (i + 1 < 250):
        plt.subplot(5, 1, 1)
        plt.plot(A_scaled, frequency)
        plt.ylabel('vowel /a/')
        
    # Vowel /i/: frames 301 to 349
    if (i + 1 > 300) and (i + 1 < 350):
        plt.subplot(5, 1, 2)
        plt.plot(A_scaled, frequency)
        plt.ylabel('vowel /i/')

    # Vowel /u/: frames 411 to 459
    if (i + 1 > 410) and (i + 1 < 460):
        plt.subplot(5, 1, 3)
        plt.plot(A_scaled, frequency)
        plt.ylabel('vowel /u/')
        
    # Vowel /e/: frames 501 to 549
    if (i + 1 > 500) and (i + 1 < 550):
        plt.subplot(5, 1, 4)
        plt.plot(A_scaled, frequency)
        plt.ylabel('vowel /e/')

    # Vowel /o/: frames 641 to 689
    if (i + 1 > 640) and (i + 1 < 690):
        plt.subplot(5, 1, 5)
        plt.plot(A_scaled, frequency)
        plt.ylabel('vowel /o/')
        
    # 2. Pitch estimation (same as ts05_01_pitch_frame)
    T0, Amax, corr_0 = ts06_pitch(sp_fr, Fs=Fs) 
    
    frame_pitch.append(T0)
    frame_pow.append(np.sum(sp_fr**2))   

# Finalize Figure 3
for j in range(5):
    plt.subplot(5, 1, j+1)
    plt.xlabel('A(k) (Scaled)')
    plt.grid(True)

plt.tight_layout()
plt.suptitle(f"LPC Spectra of Vowels in {spfile}", y=1.02)
plt.subplots_adjust(top=0.95)

# --- Pitch Analysis Plots (Same as ts05_01) ---
frame_pitch = np.array(frame_pitch)
tp = np.arange(1, num_frame + 1) # Time points for frames (1-based index)

plt.figure(2, figsize=(10, 8))

# Subplot 1: Speech signal
plt.subplot(4, 1, 1)
plt.plot(sp)
plt.ylabel('speech')
plt.title(f"Pitch Analysis for {spfile}")

# Subplot 2: Pitch interval (T0 in samples)
plt.subplot(4, 1, 2)
plt.plot(tp, frame_pitch)
plt.ylabel('pitch interval (samples)')

# Subplot 3: Pitch frequency (Hz)
pitch_freq = ts06_pitch_freq(frame_pitch)
pitch_freq_hz = pitch_freq * Fs 

plt.subplot(4, 1, 3)
plt.plot(tp, pitch_freq_hz)
plt.ylabel('pitch freq. (Hz)')

# Subplot 4: Pitch frequency with median filter (Hz)
pitch_freq2_hz = ts06_median(pitch_freq_hz) 

plt.subplot(4, 1, 4)
plt.plot(tp, pitch_freq2_hz)
plt.ylabel('pitch freq. with median filt. (Hz)')
plt.xlabel('Frame Number')

plt.tight_layout()
plt.show()