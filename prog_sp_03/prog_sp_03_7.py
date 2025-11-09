# prog_sp_03-6.py LPC analysis
import numpy as np
import scipy.io.wavfile as wavfile
# Import lfilter and freqz from scipy.signal
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt

from prog_sp_03.prog_sp_03_4 import levinson_durbin_lpc

# run with command: python -m prog_sp_03.prog_sp_03_7



# Clear all and close all (equivalent to MATLAB's commands)
plt.close('all')

# Define constants
AUDIO_FILE = 'aeiou.wav'
START_SAMPLE = 9500
END_SAMPLE = 9799

# --- read data ---
try:
    # Read a wave file. wavfile.read returns (sampling_rate, data)
    fs, sp = wavfile.read(AUDIO_FILE)
except FileNotFoundError:
    print(f"Error: The file '{AUDIO_FILE}' was not found. Please ensure it is in the current directory.")
    exit()

# Convert integer data to float and normalize if necessary for processing
if sp.dtype.kind in np.typecodes['AllInteger']:
    # Convert to float
    sp = sp.astype(float)
    # Note: Normalization is often required when plotting or performing math on raw audio data
    # but since the original MATLAB code didn't explicitly normalize after reading,
    # we'll proceed with float conversion of the raw values.

# Plot the selected segment of the original signal
sample_indices = np.arange(START_SAMPLE, END_SAMPLE + 1)
y = sp[START_SAMPLE:END_SAMPLE + 1] # Extract the segment (Python slicing is exclusive at the end)

plt.figure(1)
# Plot with relative indices for the x-axis, similar to MATLAB's plot(indices, data)
plt.plot(sample_indices, y)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Original Signal Segment (y)')
plt.grid(True)

# Calculate power of the signal segment y
# MATLAB: y'*y -> Python: y.T @ y or np.sum(y**2)
pow_y = np.sum(y**2)
print('pow_y')
print(pow_y)

# --- Q.1 LPC analysis ---
p = 10 # LPC order
# Use the custom levinson_durbin_lpc function defined earlier
try:
    a, er = levinson_durbin_lpc(y, p)
except NameError:
    print("Error: The 'levinson_durbin_lpc' function is not defined. Please ensure the cell defining it is executed.")
    exit()


# Plot the magnitude frequency response H(z) = 1/A(z)
plt.figure(2)
# The MATLAB command freqz([1], a) plots 1/A(z).
w, h = freqz([1], a, worN=512, fs=fs) # fs=fs is used to get frequency in Hz, otherwise normalized
plt.plot(w * fs / (2 * np.pi), 20 * np.log10(abs(h))) # Convert to dB and plot against Hz
plt.title('Frequency Response of the Inverse Filter 1/A(z)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)

# Plot the pole-zero map
plt.figure(3)
# zplane([1], a) plots the poles of 1/A(z), which are the roots of A(z).
# Use the custom zplane function defined elsewhere in the notebook
# Ensure the custom zplane function is defined in a cell above this one and executed.
try:
    zplane([1], a, title='Pole-Zero Plot of 1/A(z)')
except NameError:
    print("Error: The 'zplane' function is not defined. Please ensure the cell defining 'zplane' is executed.")

plt.grid(True)


# --- Q.2 residual signal ---
plt.figure(4)
# Calculate the residual signal (prediction error)
# Filter H_inv(z) = A(z)/1. MATLAB: res = filter(a, [1], y)
# Python lfilter(b, a, x): b=numerator (feedforward), a=denominator (feedback)
res = lfilter(a, [1], y)

# Plot the residual signal segment
# The original MATLAB code used the indices 9500:9799 for the x-axis,
# but the plot data is the residual 'res' which has a length of 300.
# We'll use the original indices for the x-axis for similarity.
plt.plot(sample_indices, res)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Residual Signal (res)')
plt.grid(True)

# Calculate power of the residual signal
# MATLAB: res'*res -> Python: np.sum(res**2)
pow_res = np.sum(res**2)
print('pow_res')
print(pow_res)

# Display all plots
plt.show()

# ----------------------------------------------------------------------
# --- Commented-out MATLAB Code (for reference, not executed) ---
# The following functions are not converted to executable Python
# as they were commented out in the original MATLAB code.
# They primarily deal with resynthesis and excitation generation.
'''
# --- residual signal (resynthesis) ---
# figure(5)
# y2 = filter([1],a,res)
# plot(9500:9799,y2)
# # sound(y2,8000)
# # sound(res,8000)

# # Periodic excitation signal 'c' (pulse train)
# # c = zeros(8000,1)
# # c(1:70:8000)=0.05
# c = np.zeros(8000)
# c[::70] = 0.05

# # figure(6)
# # y3 = filter([1],a,c[0:300]) # Resynthesis with pulse excitation
# # plot(9500:9799,y3)

# # figure(7)
# # plot(9500:9799,c[0:300]) # Plot the segment of the pulse train
# # y4 = filter([1],a,c)
# # sound(y4,8000)
'''
# ----------------------------------------------------------------------