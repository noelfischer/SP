import numpy as np

def ts06_pitch_freq(p_t):
    """
    Converts pitch interval (period in samples) to pitch frequency (Hz).
    
    :param p_t: numpy array of pitch intervals (in samples).
    :return: numpy array of pitch frequencies (in Hz).
    """
    p_t = np.array(p_t).flatten()
    n = len(p_t)
    p_f = np.zeros(n)

    # 1/p_t(i) for non-zero pitch intervals, 0 for zero (unvoiced)
    p_f[p_t != 0] = 1.0 / p_t[p_t != 0]
    
    # NOTE: The pitch period p_t is in samples (e.g., T0).
    # To get the frequency in Hz, the result should be multiplied by Fs (sampling frequency).
    # However, based on how the original MATLAB function is written (1/p_t), 
    # and the common convention in speech processing where pitch interval is often 
    # reported in samples and frequency in Hz, I'll follow the literal 1/p_t.
    # The calling scripts, like ts06_vocoder.m, use the output of this function 
    # directly for plotting, which suggests the *actual* frequency conversion 
    # F0 = Fs / T0 is likely intended later or implicitly handled by the context 
    # of the caller. For now, I'll keep the literal '1/p_t' as per the source.
    # The ts05_01_pitch_frame.m script, for example, plots 1/T0.
    # If the user expects Hz, a multiplication by Fs (8000) is necessary.
    
    return p_f