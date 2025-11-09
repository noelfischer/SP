import numpy as np
from scipy.signal import correlate, lfilter
from lpc_autocorr import lpc_autocorr # Not directly used but often helpful for LPC in non-standard ways

def ts06_pitch(spw, Fs=8000):
    """
    Pitch estimation based on the autocorrelation of the LPC residual.

    :param spw: numpy array, the speech frame (windowed).
    :param Fs: int, the sampling frequency (used for pitch search range).
    :return: T0 (int), Amax (float), c0 (float)
             T0: Pitch period in samples (0 if unvoiced)
             Amax: Max normalized autocorrelation value
             c0: Residual variance (autocorrelation at lag 0)
    """

    # Pitch search range: [60Hz, 300Hz] for Fs=8000Hz
    # Min Period: Fs / 300 = 8000 / 300 = 26.67 -> IP_min = 27 (or 26 as in MATLAB)
    # Max Period: Fs / 60 = 8000 / 60 = 133.33 -> IP_max = 133
    IP_min = 26 # [samples]
    IP_max = 133 # [samples]
    N_alf = 10 # LPC order

    # 1. LPC analysis
    # NOTE: scipy.signal.lpc returns [1, -a1, -a2, ...] for 'covariance' and 'prony' methods,
    # or [a0, a1, a2, ...] where a0 is not necessarily 1 for 'burg' (default).
    # MATLAB's lpc returns [1, a1, a2, ...], so we ensure we get the MATLAB format.
    # The 'prony' method is often the closest equivalent for standard LPC.
    # For a frame-based analysis like this, the 'prony' method works well.
    # The residual variance (sigma_square) is the second output in MATLAB.
    # In scipy's lpc, the first output is 'a', the second is 'e' (prediction error).
    # The total squared error is sum(e**2) / len(e). The variance of the residual is sigma_square.

    # 1. LPC analysis
    # [alf, sigma_square] = lpc(spw,10) is replaced by lpc_autocorr
    alf, sigma_square = lpc_autocorr(spw, N_alf)

    # 2. LPC residual
    # res = filter(alf,1,spw)
    # In Python, this is filtering the signal `spw` with transfer function 1/A(z),
    # where A(z) is the denominator (coefficients `alf`).
    # H(z) = B(z) / A(z). Here B(z) = 1.
    res = lfilter(alf, 1, spw)

    # 3. Autocorrelation
    # c = xcorr(res, IP_max);
    # Note: `correlate` in SciPy/NumPy needs the `mode='full'` for MATLAB's `xcorr`.
    # The max lag is IP_max, so the resulting array is 2*IP_max + 1 long.
    c = correlate(res, res, mode='full')

    # c0 = c(IP_max+1); (autocorrelation at lag 0)
    # The center of the full correlation array (lag 0) is at index len(res) - 1.
    c0_index = len(spw) - 1
    c0 = c[c0_index] # Residual power/variance

    # Cxx=c(IP_max+1:2*IP_max+1)/c0; (normalized autocorr. from lag 0 up to lag IP_max)
    # We want lags from 0 up to IP_max.
    # In the full correlation array `c`, this corresponds to indices:
    # [c0_index] to [c0_index + IP_max] (inclusive)
    # Python slice: c[c0_index : c0_index + IP_max + 1]
    Cxx_full = c[c0_index : c0_index + IP_max + 1]

    # Handle the case where c0 is near zero to avoid division by zero
    if c0 == 0:
        Cxx = Cxx_full
    else:
        Cxx = Cxx_full / c0 # Normalized autocorrelation

    # Cxx(1:26)=0; (Set lags 0 up to 25 to 0)
    # In Python, this is indices 0 up to 25.
    Cxx[0:IP_min] = 0

    # 4. Find max pitch and V/UV decision
    # [Amax,Imax]=max(Cxx);
    Imax = np.argmax(Cxx) # Index of the maximum (pitch period in samples)
    Amax = Cxx[Imax]      # Value of the maximum

    # T0 is the index (Imax) - 1 (because Cxx index 0 is T=0, index 1 is T=1, etc.)
    # The MATLAB index is 1-based. Imax is the 1-based index. The lag is Imax - 1.
    # In Python, Imax is the 0-based index. The lag is Imax.
    # Imax=1 (Python index) corresponds to lag=1 (1 sample period).
    # The range is [IP_min, IP_max]. Since we set Cxx[0:IP_min-1] to 0,
    # Imax will be at least IP_min.

    # The pitch period T0 (in samples) is simply Imax
    T0_candidate = Imax

    # V/UV condition
    if (Amax > 0.18) or (c0 > 0.01):
       T0 = T0_candidate
    else:
       T0 = 0

    # Return Amax and c0 (residual variance)
    return T0, Amax, c0