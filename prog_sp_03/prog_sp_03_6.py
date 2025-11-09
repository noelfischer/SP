# prog_sp_03-6.py inverse filter

import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# Clear all (equivalent to MATLAB's clear all and close all in a script context)
plt.close('all') 

# Define filter coefficients
coef_b = np.array([1])
coef_a = np.array([1, -1, 0.16])

# Read a wave file (assuming 'aiueo3_08k.wav' is in the current directory)
# wavfile.read returns the sampling rate (fs) and the data (y)
try:
    fs, y = wavfile.read('aeiou.wav')
except FileNotFoundError:
    print("Error: The file 'aiueo3_08k.wav' was not found. Please ensure it is in the current directory.")
    exit()

# If the audio data is integer type, convert it to float for filtering
if y.dtype.kind in np.typecodes['AllInteger']:
    y = y.astype(float) / np.max(np.abs(y)) # Normalize to [-1, 1] range

# Inverse filtering (calculate residual signal y -> x)
# MATLAB: x = filter(coef_a, coef_b, y)  -> Python: x = lfilter(coef_a, coef_b, y)
# NOTE: The MATLAB filter function's arguments for 'a' and 'b' are swapped when converting to scipy.signal.lfilter 
# for IIR filters, but for the given call x = filter(coef_a, coef_b, y) where y is the input and x is the output,
# the equivalent transfer function is H(z) = B(z)/A(z) = coef_b(z)/coef_a(z).
# For inverse filtering (y -> x), the filter is G(z) = 1/H(z) = A(z)/B(z) = coef_a(z)/coef_b(z).
# In scipy.signal.lfilter(b, a, x), 'b' is the numerator (feedforward, MA part) and 'a' is the denominator (feedback, AR part).
# So, the inverse filter with transfer function A(z)/B(z) uses A as 'b' (numerator) and B as 'a' (denominator).
x = lfilter(coef_a, coef_b, y) 

# Filtering (resynthesis x -> y2)
# The resynthesis filter is H(z) = B(z)/A(z).
# It uses B as 'b' (numerator) and A as 'a' (denominator).
y2 = lfilter(coef_b, coef_a, x)

# Plotting
# MATLAB: figure(1); hold on; subplot(3,1,1), plot(...)
plt.figure(1)

# Set the sample range for plotting (MATLAB: 8000:8511 -> Python: 8000:8512 for a length of 512)
start_sample = 8000
end_sample = 8512
sample_index = np.arange(start_sample, end_sample)
plot_length = end_sample - start_sample

# 1st Subplot: Original signal (y)
plt.subplot(3, 1, 1)
# Plot only the selected segment, indexed from 0 to plot_length for the x-axis
plt.plot(np.arange(plot_length), y[start_sample:end_sample], color='r', linestyle='-', linewidth=1)
plt.grid(True)
# Set axis limits
plt.axis([0, plot_length, -0.4, 0.4]) 
plt.ylabel('y')
plt.title('Original Signal, Residual, and Resynthesized Signal')

# 2nd Subplot: Residual signal (x)
plt.subplot(3, 1, 2)
plt.plot(np.arange(plot_length), x[start_sample:end_sample], color='b', linestyle='-', linewidth=1)
plt.grid(True)
# Set axis limits
plt.axis([0, plot_length, -0.4, 0.4])
plt.ylabel('x')

# 3rd Subplot: Resynthesized signal (y2)
plt.subplot(3, 1, 3)
plt.plot(np.arange(plot_length), y2[start_sample:end_sample], color='k', linestyle='-', linewidth=1)
plt.grid(True)
# Set axis limits
plt.axis([0, plot_length, -0.4, 0.4])
plt.ylabel('y2')
plt.xlabel('Sample Index (from 8000 to 8511)')

# Adjust layout to prevent subplot titles/labels from overlapping
plt.tight_layout()

# Display the plot
plt.show()
