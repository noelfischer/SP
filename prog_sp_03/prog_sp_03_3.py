# prog_sp_03-3.py AR filter
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# clear all;
# close all;
plt.close('all') # Equivalent to close all;

# Define filter coefficients
coef_b = np.array([1])
coef_a = np.array([1, -1, 0.16])

# Input signal
x = np.array([1, 0, 2, 0, 0, 3, 0, 1])

# Filter the signal
# [y zf] = filter(coef_b, coef_a, x);
# Equivalent to MATLAB's filter function. zf (final states) is returned but not used later.
y = signal.lfilter(coef_b, coef_a, x, zi=None)

# Apply the inverse filter
# x2 = filter(coef_a, coef_b, y)
# The second filter uses the denominator (coef_a) as the numerator and vice-versa.
# This operation is equivalent to the inverse of the first filter if the system is stable.
x2 = signal.lfilter(coef_a, coef_b, y)

# Plotting
# figure(1);
plt.figure(1)

# hold on; # Not strictly necessary in Matplotlib for subplots but a good practice.

# subplot(3,1,1), stem(x,'linewidth',2); grid on;
plt.subplot(3, 1, 1)
plt.stem(x, linefmt='C0-', markerfmt='C0o', basefmt='k-')
plt.title('Input Signal (x)')
plt.grid(True)

# subplot(3,1,2), stem(y,'linewidth',2); grid on;
plt.subplot(3, 1, 2)
plt.stem(y, linefmt='C1-', markerfmt='C1o', basefmt='k-')
plt.title('Filtered Signal (y)')
plt.grid(True)

# subplot(3,1,3), stem(x2,'linewidth',2); grid on;
plt.subplot(3, 1, 3)
plt.stem(x2, linefmt='C2-', markerfmt='C2o', basefmt='k-')
plt.title('Inverse Filtered Signal (x2)')
plt.grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# --- Code in the commented block (optional visualizations) ---
# %{
# figure(2);
# hold on;
# freqz(coef_b,coef_a);

# figure(3);
# zplane(coef_b,coef_a);
# %}

# Equivalent for freqz (Frequency Response Plot)
# Note: SciPy's freqz returns data, plotting requires Matplotlib.
plt.figure(2)
w, h = signal.freqz(coef_b, coef_a)
fig, ax = plt.subplots(2, 1)
ax[0].plot(w/(2*np.pi), 20 * np.log10(abs(h)))
ax[0].set_ylabel('Magnitude [dB]')
ax[0].set_title('Frequency Response (Magnitude)')
ax[0].grid()
ax[1].plot(w/(2*np.pi), np.unwrap(np.angle(h)))
ax[1].set_xlabel('Frequency [normalized to $f_s$]')
ax[1].set_ylabel('Phase [radians]')
ax[1].set_title('Frequency Response (Phase)')
ax[1].grid()
plt.tight_layout()

# Equivalent for zplane (Pole-Zero Plot)
# Note: Matplotlib-based zplane implementations are often custom utility functions
# or available in specialized libraries (like 'pydae'). SciPy does not have a
# direct zplane function. The code below shows how to get the poles/zeros:
zeros = np.roots(coef_b)
poles = np.roots(coef_a)
print(f"\nFilter Zeros: {zeros}")
print(f"Filter Poles: {poles}")

# Display all figures
plt.show()

# Print the resulting arrays for verification
print("\nOutput Arrays:")
print(f"y = {y}")
print(f"x2 = {x2}")
