# prog_sp_03-4.py LPC analysis and plot
import numpy as np
#import scipy.io.wavfile as wavfile
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

def levinson_durbin_lpc(y_signal, order):
    """
    Function to calculate LPC coefficients and prediction error power using the Levinson-Durbin method.
    Args:
        y_signal (np.ndarray): Input signal (1D array).
        order (int): LPC order (p).

    Returns:
        tuple: (a, err_power)
            a (np.ndarray): LPC coefficients [1, a1, a2, ..., ap].
            err_power (float): Prediction error power.
    """
    n = len(y_signal)

    #--- cal. auto correlation function
    R = np.zeros(order + 1)
    for i in range(order + 1):
        # Unbiased or biased auto-correlation calculation. Using biased (dividing by n)
        # to match typical LPC implementations.
        #R[i] = np.sum(y_signal[:n-i] * y_signal[i:]) / n # unbiased
        R[i] = np.sum(y_signal[:n-i] * y_signal[i:])      # biased (simple and matched to FFT spect.)

    #--- if signal power is very small, set output lpc coeff. and err_power = 0.
    if R[0] <= 1e-9:
        lpc_coeffs = np.zeros(order + 1)
        lpc_coeffs[0] = 1.0
        return lpc_coeffs, 0.0

    a = np.zeros(order + 1)
    a_prev = np.zeros(order + 1)
    a[0] = 1.0
    err_power = R[0] # Initial prediction error power E_0 = R(0)

    # Levinson-Durbin algorithm
    for m in range(1, order + 1):
        # Calculate reflection coefficient K_m
        numerator = R[m]
        # Note the coefficient indices to align with MATLAB's lpc function.
        # a[j] here corresponds to a_{j}^{(m-1)}
        for j in range(1, m):
            numerator += a[j] * R[m-j] # Uses the current (m-1) step's 'a' coefficients

        if err_power <= 1e-9 * R[0]: # If prediction error power becomes very small
            K_m = 0
        else:
            K_m = -numerator / err_power

        # Update coefficients a_j^(m)
        # Temporarily save the current 'a' values before updating
        for j in range(1, m):
            a_prev[j] = a[j]

        a[m] = K_m # Set a_m^(m)

        for j in range(1, m):
            a[j] = a_prev[j] + K_m * a_prev[m-j]  # a_j^(m) = a_j^(m-1) + K_m * a_{m-j}^{(m-1)}

        # Update prediction error power E_m
        err_power *= (1.0 - K_m**2)
        if err_power < 0:  # Can become negative due to floating-point error
            err_power = 0

    return a, err_power

def zplane(b, a, title='Pole-Zero Plot'):
    """
    Plots the pole-zero placement of a digital filter.
    b: Numerator coefficients
    a: Denominator coefficients
    """
    # Threshold to treat very small coefficients as zero
    threshold = 1e-9
    a = np.array(a)
    b = np.array(b)
    # Set coefficients below threshold to zero
    a[np.abs(a) < threshold] = 0
    b[np.abs(b) < threshold] = 0

    try:
        # Consider cases where tf2zpk might produce complex Infinity or NaN
        zeros, poles, k = signal.tf2zpk(b, a)
        # Remove Infinity or NaN
        zeros = zeros[np.isfinite(zeros)]
        poles = poles[np.isfinite(poles)]

    except ValueError as e:
        print(f"Error in tf2zpk: {e}")
        print(f"b: {b}, a: {a}")
        # Exit without plotting if an error occurs
        return

    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    unit_circle = Circle((0,0), 1, color='grey', fill=False, linestyle='--')
    ax.add_patch(unit_circle)

    # Plot zeros (displayed as o)
    if len(zeros) > 0:
        # Handle repeated zeros
        distinct_zeros, zero_counts = np.unique(zeros.round(decimals=6), return_counts=True) # Round then find unique
        for z, count in zip(distinct_zeros, zero_counts):
            plt.plot(np.real(z), np.imag(z), 'bo', markersize=10, fillstyle='none', label='Zeros' if z == distinct_zeros[0] else "", alpha=0.7)
            if count > 1:
                plt.text(np.real(z), np.imag(z), f' {count}', fontsize=12, ha='left', va='bottom')

    # Plot poles (displayed as x)
    if len(poles) > 0:
         # Handle repeated poles
        distinct_poles, pole_counts = np.unique(poles.round(decimals=6), return_counts=True) # Round then find unique
        for p_val, count in zip(distinct_poles, pole_counts):
            plt.plot(np.real(p_val), np.imag(p_val), 'rx', markersize=10, label='Poles' if p_val == distinct_poles[0] else "", alpha=0.7)
            if count > 1:
                plt.text(np.real(p_val), np.imag(p_val), f' {count}', fontsize=12, ha='left', va='bottom')


    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title(title)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.set_xlim([-1.1, 1.1]) # Match the circle boundary
    ax.set_ylim([-1.1, 1.1]) # Match the circle boundary
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.legend()
    ax.axis('equal') # Set aspect ratio to equal


# --- Main Processing ---
if __name__ == "__main__":

    # input speech wav file (16kHz sampling)
    wav_filename = './aeiou16.wav'

    # LPC analysis order
    p = 16

    sp, sample_rate = sf.read(wav_filename)        # read the wav speech
    print(f"sample_rate={sample_rate}")
    print(f"sp size ={len(sp)}")

    # extract an analysis frame. 640 samples = 40 ms
    start_idx = 9500
    end_idx   = start_idx + 640 - 1

    y0 = sp[start_idx:end_idx+1]                    # 1 frame speech data
    print(f"y0 size ={len(y0)}")
    time_axis_plot = np.arange(start_idx,end_idx+1)  # time samples

    # win = np.hanning(len(y0))
    win = np.hamming(len(y0))    #use hamming window
    y = win * y0                 #multiply the window

    #-- fft
    fft_size = 2048 # Used to match the number of plot points for FFT and LPC spectrum
    Y = np.fft.fft(y, fft_size) # zero padding to fft_size
    Y_magnitude = np.abs(Y[:fft_size // 2 + 1])
    print("Y_magnitude.size=",len(Y_magnitude))

    #--- LPC analysis by levinson_durbin method
    a, er = levinson_durbin_lpc(y, p)
    print(f"LPC order (p): {p}")
    print(f"LPC coefficients (a): {a}")
    print(f"Prediction Error Power (er): {er}")

    #- LPC model gain G
    lpc_gain = np.sqrt(er)
    print(f"LPC gain (=sqrt(er)): {lpc_gain}")

    # w is the normalized frequency (0 to pi, pi is the Nyquist frequency)
    # h_response is the complex frequency response of 1/A(z)
    w, h_response = signal.freqz([1.0], a, worN=fft_size//2+1) # frequency, 1/A(z) response

    # Multiply the frequency response by the gain (square root of the prediction error power)
    # This is the standard way to scale the LPC spectrum to match the magnitude of the original signal's spectrum.
    scaled_h_response = lpc_gain * h_response

    # --- Removed non-standard gain adjustment ---
    # lpc_power = np.sum(np.abs(scaled_h_response)**2)
    # Y_power = np.sum(Y_magnitude**2)

    # # Adjust the gain so the total power of the LPC envelope roughly matches the signal FFT power
    # if Y_power > lpc_power:
    #     gain_offset = np.sqrt(Y_power - lpc_power)
    #     scaled_h_response = gain_offset * scaled_h_response
    # else:
    #     gain_offset = np.sqrt(lpc_power - Y_power)
    #     Y_magnitude = gain_offset * Y_magnitude
    # --- End of removed non-standard gain adjustment ---


    # Amplitude (dB)
    lpc_magnitude_db = 20 * np.log10(np.abs(scaled_h_response) + np.finfo(float).eps)
    print("lpc_magnitude_db.size=",len(lpc_magnitude_db))
    Y_magnitude_db = 20 * np.log10(Y_magnitude + np.finfo(float).eps)
    print("Y_magnitude_db.size=",len(Y_magnitude_db))

    # normalized freq axis
    fft_freq_norm = np.linspace(0, 1, fft_size // 2 + 1)

    #--- plot a frame of the input speech
    plt.figure(1)
    plt.plot(time_axis_plot, y0)
    plt.title(f'Original Signal Segment (Fs={sample_rate}Hz)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("prog_onsei_5-1(1).png")

    #--- plot the LPC spectrum and FFT spectrum
    plt.figure(2)
    # x axis: normalized frequency [0, 1], Nyquist freq. = 1
    plt.plot(w / np.pi, lpc_magnitude_db, label='Scaled LPC Envelope')
    plt.plot(fft_freq_norm, Y_magnitude_db, label='Windowed Signal FFT', alpha=0.7) #alpha: transparency
    plt.title(f'Spectrum Comparison (Fs={sample_rate}Hz)')
    plt.xlabel('Normalized Frequency (x $\\pi$ rad/sample, 1.0 = Nyquist)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.legend() # Display legend
    # adjust Y axis range
    min_db = np.min(Y_magnitude_db)
    plt.ylim([min(min_db - 10, -60), max(np.max(lpc_magnitude_db) + 10, np.max(Y_magnitude_db) + 10, 0)])
    plt.savefig("prog_onsei_5-1(2).png")

    #--- plot phase spectrum
    plt.figure(3)
    phase_response = np.angle(h_response) #radian
    plt.plot(w / np.pi, phase_response)
    plt.title(f'LPC Spectrum Envelope (1/A(z)) Phase (Fs={sample_rate}Hz)')
    plt.xlabel('Normalized Frequency (x $\\pi$ rad/sample, 1.0 = Nyquist)')
    plt.ylabel('Phase (radians)')
    plt.grid(True)
    plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    plt.savefig("prog_onsei_5-1(3).png")

    #--- plot pole-zero
    # zplane([MAcoef(=b)], [ARcoef(=a)])
    zplane([1.0], a, title=f'Pole-Zero Plot for 1/A(z) (Fs={sample_rate}Hz)')
    plt.figure(4)
    plt.savefig("prog_onsei_5-1(4).png")

    #--- plot residual signal
    res = signal.lfilter(a, [1.0], y)    # res = Y(Z)A(z)
    plt.figure(5)
    plt.plot(time_axis_plot, res)
    plt.title(f'Residual Signal (e) (Fs={sample_rate}Hz)')
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("prog_onsei_5-1(5).png")

    plt.tight_layout()  # adjustment of figure intervals
    plt.show()

    #EOF