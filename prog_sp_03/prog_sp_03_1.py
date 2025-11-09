
# prog_sp_03-1.py  xcorr
import numpy as np
import matplotlib.pyplot as plt

# --- function ---
def my_xcorr(x, nc):
    """
    my_xcorr: my auto-correlation function
    c_out = my_xcorr(x, nc)

    Args:
        x (numpy.ndarray): input
        nc (int): maximum lag (delay)

    Returns:
        numpy.ndarray: auto-correlation function
    """
    N = len(x)
    # array of the corrlation lag 0 to lag nc
    c = np.zeros(nc + 1) 

    # cal correlation from lag m m = 0 to nc
    for m in range(nc + 1):
        sqsum = 0.0
        for n in range(N - m):
            sqsum += x[n] * x[n + m]
        c[m] = sqsum

    # Python c_out = np.concatenate([c[::-1], c[1:nc+1]]) 
    c_out = np.concatenate([c[::-1], c[1:]])

    return c_out

# --- main program ---

# x = np.array([1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0, 1, 2, 3, 0, 0])
x = np.array([2, -2, 3, 0, 0, 2, -2, 3, 0, 0, 2, -2, 3, 0, 0, 2, -2, 3, 0, 0, 2, -2, 3, 0, 0])

nc = 20
lags = np.arange(-nc, nc + 1) # lag array for plot

# --- Figure 1
plt.figure(1)
# NumPy correlate 'full'mode: calculate (2*N - 1) point values
# NumPy correlation function is a reverse of MATLAB xcorr
c_full = np.correlate(x, x, mode='full')
# extract lag -nc to nc values
start_idx = len(x) - 1 - nc
end_idx = len(x) - 1 + nc + 1
c = c_full[start_idx:end_idx]

plt.stem(lags, c, basefmt=" ")
plt.title('Auto-correlation (Raw)')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.grid(True)

# --- Figure 2: normalized auto-correlatin function ('coef'option)
plt.figure(2)

cc_full = np.correlate(x, x, mode='full')
# normalized factor = lag 0 correlation value (cc_full N-1th element)
norm_factor = cc_full[len(x) - 1]
# normalizetion 
cc_norm = cc_full / norm_factor
# lag -nc to nc values
cc = cc_norm[start_idx:end_idx]

plt.stem(lags, cc, basefmt=" ")
plt.title('Auto-correlation (Coefficient)')
plt.xlabel('Lag')
plt.ylabel('Normalized Correlation')
plt.grid(True)

# ---
plt.figure(3)
my_c = my_xcorr(x, nc)

plt.stem(lags, my_c, basefmt=" ")
plt.title('Auto-correlation (my_xcorr)')
plt.xlabel('Lag')
plt.ylabel('Correlation')
plt.grid(True)

plt.show()
