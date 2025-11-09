# prog_sp_03-9.py work_pitch
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import correlate
import matplotlib.pyplot as plt

# Clear all and close all (equivalent to MATLAB's commands)
plt.close('all') 

# Define constants
AUDIO_FILE = 'aeiou.wav'
START_SAMPLE = 10000
END_SAMPLE = 10299

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

# --- Calculate power of y ---
# MATLAB: y'*y -> Python: np.sum(y**2)
pow_y = np.sum(y**2)
print('pow_y')
print(pow_y)

# --- Autocorrelation analysis ---
plt.figure(2)
# MATLAB: xc_y = xcorr(y, length(y), 'coef')
# This calculates the full autocorrelation and normalizes by R[0] 
# (the energy of the segment, R[0] = np.sum(y**2))

# 1. Calculate full correlation
xc_y_full = correlate(y, y, mode='full')

# 2. Normalize by R[0] to get the 'coef' (coefficient) option
R0 = xc_y_full[segment_length - 1] # R[0] is at the middle of the 'full' result
xc_y_norm = xc_y_full / R0

# Define the search range for the pitch period
# MATLAB search range: length(y)+20:length(y)+length(y)
# This corresponds to lags from 20 up to length(y)-1 in the positive lag array.
search_start_lag = 20
# Extract the positive lag part (from lag 0 to length(y)-1)
xc_y_positive_lags = xc_y_norm[segment_length - 1:]

# Search segment (lags 20 to end)
search_segment = xc_y_positive_lags[search_start_lag:]

# Find the maximum (py) and its index (ipy) in the search segment
ipy = np.argmax(search_segment)
py = search_segment[ipy]

# Adjust index to be the actual lag value
# The lag is ipy (index in search_segment) + search_start_lag
ipy = ipy + search_start_lag

# Plot the full autocorrelation result (all lags)
lags = np.arange(-segment_length + 1, segment_length)
plt.plot(lags, xc_y_norm)
plt.title('Normalized Autocorrelation of Segment (y)')
plt.xlabel('Lag (samples)')
plt.ylabel('Normalized Autocorrelation')
plt.grid(True)

# Display the maximum value (py) and its lag (ipy), which is the pitch period estimate
print(f"py: {py}")
print(f"ipy: {ipy}")

# Sound commands are commented out in the original MATLAB code
# sound(y2, 8000)
# sound(res, 8000)

# Display all plots
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------
# --- Commented-out MATLAB Code (for reference, not executed) ---
'''
# Excitation signal generation and resynthesis (requires 'a' coefficients from LPC, 
# which are not calculated in this version of the script)

# c = zeros(8000,1);
# c(1:70:8000)=0.05;
# c = np.zeros(8000)
# c[::70] = 0.05 

# figure(6);
# y3 = filter([1],a,c[0:300]); 
# plot(9500:9799,y3);

# figure(7);
# plot(9500:9799,c[0:300]); 

# y4 = filter([1],a,c);
# # sound(y4,8000);
'''
# ----------------------------------------------------------------------