import numpy as np
from scipy.signal import medfilt

def ts06_median(xi):
    """
    Applies a median filter of size 5 to the input signal.
    
    :param xi: numpy array, the input signal.
    :return: numpy array, the median-filtered output signal.
    """
    xi = np.array(xi).flatten()
    n = len(xi)
    
    # MATLAB's median filter implementation in this script uses a 5-point window 
    # with padding of two zeros at the beginning and two at the end.
    # xi2 = [0; 0; xi; 0; 0];
    # xo(i) = median(xi2(i:i+nm-1)); where nm=5
    # This results in a *causal* or *forward-looking* window for the first few points,
    # and a *non-symmetric* window for the middle.
    
    # A cleaner approach in Python using scipy's medfilt:
    # We can use scipy.signal.medfilt with a kernel size of 5.
    # By default, medfilt uses 'reflect' padding to handle edges.
    
    # The MATLAB implementation's explicit zero-padding is non-standard
    # for a typical median filter, but to replicate it exactly:
    
    xi2 = np.concatenate(([0, 0], xi, [0, 0]))
    xo = np.zeros(n)
    nm = 5
    
    for i in range(n):
        # The window in MATLAB is xi2(i:i+nm-1), where MATLAB indices are 1-based.
        # i=1 (Python 0) -> xi2[0:5]
        # i=n (Python n-1) -> xi2[n-1:n-1+5] = xi2[n-1:n+4]
        xo[i] = np.median(xi2[i : i + nm])
        
    return xo