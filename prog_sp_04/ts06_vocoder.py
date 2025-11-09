import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal.windows import hamming
from scipy.signal import lfilter
from lpc_autocorr import lpc_autocorr
from ts06_pitch import ts06_pitch
from ts06_pitch_freq import ts06_pitch_freq
from ts06_median import ts06_median

#--- constants ---
Fs =  8000 # sampling frequency [Hz]
Lf =    240 # frame length [sample]
Ls =    40 # frame shift length [sample]
N_alf = 10 # LPC analysis order

#--- read input speech file ---
# The original script had commented-out files. We'll use 'M01_mip_08k.wav'.
spfile = 'M01_mip_08k.wav'

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

# Add very small noise
sp = sp + np.random.randn(len(sp)) * 1.0e-10

# Plot original speech
plt.figure(1)
plt.plot(sp)
plt.title('Original Speech Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# initialize variables and data buffers
sp_out = []
frame_pitch = []
frame_sigma = []
frame_gain = []
frame_pow = []
z_buff = np.zeros(N_alf)
offset = 0

# Calculation for number of frames
num_frame = int(np.floor((len(sp) - Lf) / Ls)) + 1

for i in range(num_frame):

    fr_start = i * Ls
    fr_end   = i * Ls + Lf
    sp_fr = sp[fr_start : fr_end]

    if len(sp_fr) < Lf:
        break

    # LPC analysis with Hamming window
    # MATLAB's `lpc` returns [1, a1, ..., ap] and the residual variance.
    # We use `lpc` from SciPy, ensuring a MATLAB-like output.
    ai, sigma_square = lpc_autocorr(sp_fr * hamming(Lf), N_alf)
    sigma = np.sqrt(sigma_square)

    # local synthesis pitch period (in samples)
    T0, Amax, corr_0 = ts06_pitch(sp_fr, Fs=Fs)

    # Generating of excitation
    if (T0 != 0): # Voiced frame
        # Generation of voiced excitation signals (periodic pulse train)

        # Calculate pulse positions for the current Ls samples
        exc = np.zeros(Ls)
        current_ls_samples_processed = 0

        # Handle the offset from the previous frame
        if offset >= Ls:
            # Entire current frame is covered by the tail of the previous period
            exc = np.zeros(Ls)
            offset -= Ls
        else:
            # Place the first pulse at `offset` samples from the start
            pulse_index = int(offset)

            # The original MATLAB code's logic is complex due to integer arithmetic.
            # A more robust Python logic for the pulse train:
            while pulse_index < Ls:
                exc[pulse_index] = 1 # Place a pulse
                pulse_index += T0    # Move to the next pulse position

            # The new offset is the distance from the last sample of the current
            # frame (Ls-1) to the position of the next pulse.
            # If the last pulse was at `k < Ls`, the next would be at `k+T0`.
            # The remaining distance for the next frame to start from is `k+T0 - Ls`.
            if pulse_index > Ls:
                 offset = pulse_index - Ls
            else:
                 offset = 0 # Exact alignment at the end of the frame

        # Gain calculation: sigma / sqrt(Power of Excitation)
        # Power of Excitation: sum(exc**2) / Ls
        # For a normalized pulse train (pulses are 1), total power in Ls is (Ls/T0)/Ls = 1/T0
        gain = sigma / np.sqrt(1.0 / T0)
    else:
        # Generation of unvoiced excitation signals (White Gaussian noise)
        exc = np.random.randn(Ls)
        # Gain calculation: sigma / sqrt(Power of Excitation)
        # Power of Excitation: sum(exc**2) / Ls ~ 1 (for unit variance noise)
        gain = sigma
        offset = 0 # reset for subsequent voiced frames

    # Append frame statistics
    frame_pitch.append(T0)
    frame_gain.append(gain)
    frame_sigma.append(sigma)
    frame_pow.append(np.sum(sp_fr**2))

    # LPC Synthesis: H(z) = G / A(z). Use `lfilter` with B=[G] and A=ai.
    # The `z_buff` is the initial condition (filter state) for the IIR filter.
    # For `lfilter(b, a, x, zi=z_buff)`: the first output is `y`, the second is `zf` (final state).
    sp_out_fr, z_buff = lfilter([gain], ai, exc, zi=z_buff)
    sp_out.extend(sp_out_fr)

# Convert lists to numpy arrays
sp_out = np.array(sp_out)
frame_pitch = np.array(frame_pitch)
frame_gain = np.array(frame_gain)
frame_sigma = np.array(frame_sigma)
frame_pow = np.array(frame_pow)
tp = np.arange(1, num_frame + 1) # Time points for frames (1-based index)

# --- Plotting ---

# Figure 2: Output speech
plt.figure(2)
plt.plot(sp_out)
plt.title('Synthesized Speech Signal (Vocoder Output)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')

# Figure 3: Pitch, Gain, Sigma (Scaled)
plt.figure(3)
plt.plot(tp, frame_pitch, label='Pitch Interval (T0)')
plt.plot(tp, frame_gain * 500, label='Gain * 500')
plt.plot(tp, frame_sigma * 500, label='Sigma * 500')
plt.legend()
plt.title('Frame Parameters')
plt.xlabel('Frame Number')

# Figure 5: Pitch Interval
plt.figure(5)
plt.plot(tp, frame_pitch)
plt.title('Pitch Interval (T0)')
plt.xlabel('Frame Number')
plt.ylabel('T0 (samples)')

# Figure 4: Gain
plt.figure(4)
plt.plot(tp, frame_gain)
plt.title('LPC Synthesis Gain')
plt.xlabel('Frame Number')
plt.ylabel('Gain')

# Figure 6: Pitch Frequency (1/T0 * Fs)
pitch_freq = ts06_pitch_freq(frame_pitch)
pitch_freq_hz = pitch_freq * Fs
plt.figure(6)
plt.plot(tp, pitch_freq_hz)
plt.title('Pitch Frequency')
plt.xlabel('Frame Number')
plt.ylabel('F0 (Hz)')

# Figure 8: Median-Filtered Pitch Frequency
pitch_freq2 = ts06_median(pitch_freq_hz)
plt.figure(8)
plt.plot(tp, pitch_freq2)
plt.title('Median-Filtered Pitch Frequency')
plt.xlabel('Frame Number')
plt.ylabel('F0 (Hz)')

# --- Audio Output ---
# Normalize to 16-bit integer range (or float for better compatibility)
max_val = np.iinfo(np.int16).max
sp_out_int = (sp_out * max_val).astype(np.int16)
wavfile.write('x08out.wav', Fs, sp_out_int)
print("Synthesized audio written to 'x08out.wav'")

# --- Spectrograms (Requires Matplotlib >= 3.3.0 for better 'spectrogram' args) ---
# NOTE: The original MATLAB script uses the signal processing toolbox's spectrogram,
# which may have different defaults than Matplotlib's. I'll use standard Matplotlib.
# The window length is 40 samples (5ms) for figure 10, and 240 samples (30ms) for 11/12.
# The overlap (20 and 220 samples) is unusual (typically overlap < window size).

# Figure 10: Spectrogram of output (short window)
plt.figure(10)
plt.specgram(sp_out, NFFT=40, Fs=Fs, noverlap=(40-20), window=hamming(40))
plt.ylim(0, Fs/2)
plt.title('Spectrogram of Output (Short Window)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')

# Figure 11: Spectrogram of output (long window)
plt.figure(11)
plt.specgram(sp_out, NFFT=240, Fs=Fs, noverlap=(240-220), window=hamming(240))
plt.ylim(0, Fs/2)
plt.title('Spectrogram of Output (Long Window)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')

# Figure 12: Spectrogram of input (long window)
plt.figure(12)
plt.specgram(sp, NFFT=240, Fs=Fs, noverlap=(240-220), window=hamming(240))
plt.ylim(0, Fs/2)
plt.title('Spectrogram of Input (Long Window)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()