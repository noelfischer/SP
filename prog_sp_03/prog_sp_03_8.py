# prog_sp_03-8.py  pitch
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter, freqz
# from scipy.linalg import levinson # Remove this import
import matplotlib.pyplot as plt
from prog_sp_03.prog_sp_03_4 import levinson_durbin_lpc

# from zplane import zplane # Custom function for z-plane plot (often implemented separately)

# NOTE: The zplane function is not a standard part of SciPy.
# You will need to ensure a 'zplane.py' file or similar utility is available
# or use a third-party library that provides this visualization.
# For this conversion, we assume a compatible 'zplane' function is available
# and defined in another cell in this notebook.

# Clear all and close all
plt.close('all')

# Define constants
AUDIO_FILE = 'aeiou.wav'
START_SAMPLE = 9500
END_SAMPLE = 9799
LPC_ORDER = 10

# --- Read data ---
try:
    # Read a wave file. wavfile.read returns (sampling_rate, data)
    fs, sp = wavfile.read(AUDIO_FILE)
except FileNotFoundError:
    print(f"Error: The file '{AUDIO_FILE}' was not found. Please ensure it is in the current directory.")
    exit()

# Convert integer data to float for signal processing
if sp.dtype.kind in np.typecodes['AllInteger']:
    sp = sp.astype(float)

# Extract the signal segment (Python slicing is exclusive at the end)
sample_indices = np.arange(START_SAMPLE, END_SAMPLE + 1)
y = sp[START_SAMPLE:END_SAMPLE + 1]
segment_length = len(y)

# Plot the selected segment of the original signal
plt.figure(1)
plt.plot(sample_indices, y)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Original Signal Segment (y)')
plt.grid(True)

# Calculate power of the signal segment y
# MATLAB: y'*y -> Python: np.sum(y**2)
pow_y = np.sum(y**2)
print('pow_y')
print(pow_y)

# --- LPC analysis (Q.1) ---
# Use the custom levinson_durbin_lpc function defined earlier in the notebook
try:
    # Ensure the cell containing the definition of levinson_durbin_lpc is executed first.
    a, er = levinson_durbin_lpc(y, LPC_ORDER)
except NameError:
    print("Error: The 'levinson_durbin_lpc' function is not defined. Please ensure the cell defining it is executed.")
    exit()


# Plot the magnitude frequency response H(z) = 1/A(z)
plt.figure(2)
# The MATLAB command freqz([1], a) plots 1/A(z).
w, h = freqz([1], a, worN=512, fs=fs)
plt.plot(w * fs / (2 * np.pi), 20 * np.log10(np.abs(h)))
plt.title('Frequency Response of the Inverse Filter 1/A(z)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid(True)

# Plot the pole-zero map
plt.figure(3)
# zplane([1], a) plots the poles of 1/A(z), which are the roots of A(z).
# NOTE: Assumes 'zplane' function is available and defined in another cell.
try:
    zplane([1], a, title='Pole-Zero Plot of 1/A(z)')
except NameError:
    print("Error: The 'zplane' function is not defined. Please ensure the cell defining 'zplane' is executed.")

plt.grid(True)


# --- Residual signal (Q.2) ---
plt.figure(4)
# Calculate the residual signal (prediction error)
# Filter H_inv(z) = A(z)/1. MATLAB: res = filter(a, [1], y)
# Python lfilter(b, a, x): b=numerator (feedforward), a=denominator (feedback)
res = lfilter(a, [1], y)

# Plot the residual signal segment
plt.plot(sample_indices, res)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Residual Signal (res)')
plt.grid(True)

# --- Autocorrelation analysis ---
# Find pitch period estimate using autocorrelation

# 1. Autocorrelation of original signal (y)
plt.figure(8)
# MATLAB: xc_y = xcorr(y, length(y), 'coef') -> Python: np.correlate with 'full' mode and normalization
# Use 'full' mode to get lags from -L+1 to L-1 (where L=len(y)).
# Normalize by dividing by the value at lag 0 (np.sum(y**2))
xc_y_full = np.correlate(y, y, mode='full') / np.sum(y**2)

# MATLAB's 'xcorr(..., 'coef')' normalizes by R[0].
# The indices for positive lags are from length(y)-1 onwards in the full result.
xc_y = xc_y_full[segment_length-1:] # Take non-negative lags

# MATLAB search range: length(y)+20:length(y)+length(y) (in full result, which is 2*length(y)-1 long)
# Python search range: 20:length(y) (in the positive lag result)
search_start_lag = 20
search_end_lag = segment_length
search_segment_y = xc_y[search_start_lag:search_end_lag]

# Find the maximum (py) and its index (ipy) in the search segment
ipy = np.argmax(search_segment_y)
py = search_segment_y[ipy]

# Adjust index to be relative to the full positive-lag result
ipy = ipy + search_start_lag

# The MATLAB plot(xc_y) plots the full result (all lags)
plt.plot(np.arange(-segment_length + 1, segment_length), xc_y_full)
plt.title('Autocorrelation of Original Signal (y)')
plt.xlabel('Lag (samples)')
plt.ylabel('Normalized Autocorrelation')
plt.grid(True)
print(f"py: {py}")
print(f"ipy: {ipy}")

# 2. Autocorrelation of residual signal (res)
plt.figure(9)
xc_res_full = np.correlate(res, res, mode='full') / np.sum(res**2)
xc_res = xc_res_full[segment_length-1:] # Take non-negative lags

search_segment_res = xc_res[search_start_lag:search_end_lag]

# Find the maximum (pres) and its index (ipres) in the search segment
ipres = np.argmax(search_segment_res)
pres = search_segment_res[ipres]

# Adjust index
ipres = ipres + search_start_lag

# The MATLAB plot(xc_res) plots the full result (all lags)
plt.plot(np.arange(-segment_length + 1, segment_length), xc_res_full)
plt.title('Autocorrelation of Residual Signal (res)')
plt.xlabel('Lag (samples)')
plt.ylabel('Normalized Autocorrelation')
plt.grid(True)
print(f"pres: {pres}")
print(f"ipres: {ipres}")


# Calculate power of the residual signal
pow_res = np.sum(res**2)
print('pow_res')
print(pow_res)

# --- Resynthesis ---
plt.figure(5)
# Resynthesis filter H(z) = 1/A(z). MATLAB: y2 = filter([1], a, res)
y2 = lfilter([1], a, res)
plot_indices = np.arange(START_SAMPLE, END_SAMPLE + 1)
plt.plot(plot_indices, y2)
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.title('Resynthesized Signal (y2)')
plt.grid(True)

# Sound commands are commented out in the original MATLAB code
# sound(y2, 8000) -> sf.play(y2, fs) if using soundfile library
# sound(res, 8000) -> sf.play(res, fs)

# Display all plots
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# --- Commented-out MATLAB Code (for reference, not executed) ---
'''
# Excitation signal generation and resynthesis
# c = zeros(8000,1);
# c(1:70:8000)=0.05;
c = np.zeros(8000)
c[::70] = 0.05

# figure(6);
# y3 = filter([1],a,c[0:300]); # Resynthesis with pulse excitation segment
# plot(9500:9799,y3);

# figure(7);
# plot(9500:9799,c[0:300]); # Plot the segment of the pulse train

# y4 = filter([1],a,c);
# # sound(y4,8000);
'''
# ----------------------------------------------------------------------