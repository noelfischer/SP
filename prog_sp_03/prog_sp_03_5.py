# prog_sp_03-5.py test_filter_ar_sig
import numpy as np
import soundfile as sf
from scipy import signal
import matplotlib.pyplot as plt

# --- Custom zplane function (necessary as Matplotlib/SciPy doesn't include one) ---
# A basic version of zplane is included here for completeness, as requested by the MATLAB code.
def zplane(b, a, title='Pole-Zero Plot'):
    """
    Plots the pole-zero placement of a digital filter.
    b: Numerator coefficients
    a: Denominator coefficients
    """
    # Calculate zeros and poles
    try:
        zeros, poles, k = signal.tf2zpk(b, a)
    except ValueError:
        print("Warning: Could not calculate zeros/poles using tf2zpk.")
        return

    plt.figure(figsize=(6, 6))
    ax = plt.gca()

    # Plot unit circle
    unit_circle = plt.Circle((0, 0), 1, color='grey', fill=False, linestyle='--')
    ax.add_patch(unit_circle)

    # Plot zeros (o)
    if zeros.size > 0:
        ax.plot(np.real(zeros), np.imag(zeros), 'bo', fillstyle='none', markersize=8, label='Zeros')

    # Plot poles (x)
    if poles.size > 0:
        ax.plot(np.real(poles), np.imag(poles), 'rx', markersize=8, label='Poles')

    # Configure plot
    ax.set_title(title)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.grid(True)
    ax.axis('equal') # Set aspect ratio to equal
    # Set limits slightly outside the unit circle
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    if zeros.size > 0 or poles.size > 0:
        ax.legend()


# --- Main Program ---

# clear all; close all;
plt.close('all') # Close all figures

# Define filter coefficients (b: numerator, a: denominator)
coef_b = np.array([1, -1]) # Finite Impulse Response (FIR) part
coef_a = np.array([1])     # Infinite Impulse Response (IIR) part (a[0]=1 is standard)

# Load input signal x
# x = [1 0 0 0 0 0 0 0 ];
wav_filename = 'aeiou.wav'
try:
    # audioread returns data and sampling frequency (fs)
    x, fs = sf.read(wav_filename)
    print(f"Signal loaded from {wav_filename}. Sample rate: {fs} Hz")
except FileNotFoundError:
    print(f"Error: WAV file '{wav_filename}' not found. Using dummy signal.")
    x = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 1000) # Dummy signal if file is missing
    fs = 8000 # Assume 8kHz for dummy

# Apply the filter (e.g., a differentiator or high-pass filter: H(z) = 1 - z^-1)
# [y zf] = filter(coef_b, coef_a, x);
# zf (final states) is returned but not used later.
y = signal.lfilter(coef_b, coef_a, x)

# Apply the inverse filter (H_inv(z) = 1 / (1 - z^-1))
x2 = signal.lfilter(coef_a, coef_b, y)


# --- Plotting Time-Domain Signals ---

# figure(1); hold on;
plt.figure(1, figsize=(10, 8))
# Set the start and end sample indices for plotting
start_index = 8000 - 1 # Python is 0-indexed
end_index = 8511       # Python index 8511 corresponds to MATLAB index 8511
# Create a time axis for better visualization (optional but good practice)
time_axis = np.arange(start_index, end_index + 1)

# subplot(3,1,1), plot(x(8000:8511),'-r','linewidth',1); grid on;
plt.subplot(3, 1, 1)
plt.plot(time_axis, x[start_index:end_index+1], '-r', linewidth=1)
plt.title('Original Signal (x)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)

# subplot(3,1,2), plot(y(8000:8511),'-b','linewidth',1); grid on;
plt.subplot(3, 1, 2)
plt.plot(time_axis, y[start_index:end_index+1], '-b', linewidth=1)
plt.title('Filtered Signal (y)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)

# subplot(3,1,3), plot(x2(8000:8511),'-g','linewidth',1); grid on;
plt.subplot(3, 1, 3)
plt.plot(time_axis, x2[start_index:end_index+1], '-g', linewidth=1)
plt.title('Inverse Filtered Signal (x2)')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)

plt.tight_layout() # Adjust layout


# --- Plotting FFT Spectrum ---

L = 128 # FFT length
# Select the same segment for FFT
segment_x = x[start_index:end_index+1]
segment_y = y[start_index:end_index+1]

# Calculate FFTs
# yx = fft(x(8000:8511),L);
yx = np.fft.fft(segment_x, L)
# yy = fft(y(8000:8511),L);
yy = np.fft.fft(segment_y, L)

# figure(2); hold on;
plt.figure(2, figsize=(8, 5))
# Calculate frequency axis for plotting (in Hz)
freq_axis = np.linspace(0, fs/2, L//2 + 1)

# plot(2*abs(yx(1:L/2+1)),'-r'); grid on;
# Multiply by 2 because the negative frequency components are folded,
# and the DC component (first index) should not be doubled.
# Note: The MATLAB code incorrectly doubles the DC term. Standard practice is to
# double non-DC/non-Nyquist terms. Keeping the simple MATLAB-style multiplication
# for direct equivalence, but be aware it's not a standard magnitude plot in power.
# A better plot would be 20*log10(abs(Y))
plt.plot(freq_axis, 2 * np.abs(yx[:L//2 + 1]), '-r', label='FFT of x')
# plot(2*abs(yy(1:L/2+1)),'-b'); grid on;
plt.plot(freq_axis, 2 * np.abs(yy[:L//2 + 1]), '-b', label='FFT of y')

plt.title(f'Segment FFT Magnitudes (L={L}, Fs={fs} Hz)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (Non-Standard Scaling)')
plt.grid(True)
plt.legend()


# --- Plotting Frequency Response (freqz) ---

# figure(3); hold on; freqz(coef_b,coef_a);
plt.figure(3, figsize=(8, 6))
# freqz returns w (normalized frequency in radians) and h (complex response)
w, h = signal.freqz(coef_b, coef_a)

# Plot magnitude (dB)
ax1 = plt.subplot(2, 1, 1)
# Convert normalized frequency to Hz
freq_hz = w * fs / (2 * np.pi)
ax1.plot(freq_hz, 20 * np.log10(np.abs(h)))
ax1.set_title(f'Filter Frequency Response (Magnitude, Fs={fs} Hz)')
ax1.set_ylabel('Magnitude (dB)')
ax1.grid(True)

# Plot phase (radians)
ax2 = plt.subplot(2, 1, 2, sharex=ax1)
ax2.plot(freq_hz, np.angle(h))
ax2.set_title('Filter Frequency Response (Phase)')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Phase (radians)')
ax2.grid(True)

plt.tight_layout()


# --- Plotting Pole-Zero Map (zplane) ---

# figure(4); zplane(coef_b,coef_a);
plt.figure(4)
zplane(coef_b, coef_a, title=f'Pole-Zero Plot for H(z) = {coef_b[0]} - {abs(coef_b[1])}z?1')

# Show all generated figures
plt.show()

# %
# EOF