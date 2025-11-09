# qlog_midriser.py
# (Translation of qlog_midriser.m)

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

def run_log_quantizer():
    plt.close('all')
    print('--- Logarithmic quantization, midriser type (µ-law) ---')

    bit = 6
    scale = 2**(bit - 1)
    mu = 255.0  # Use float for calculations

    print(f'Quantization bit = {bit}')
    print(f'Scale = {scale}')
    print(f'Mu = {mu}')

    # read wav file
    wav_nbit = 16 # Original script hardcodes this
    try:
        x_in, wav_fs = sf.read('jikken16.wav')
    except FileNotFoundError:
        print("Error: 'jikken16.wav' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the audio file: {e}")
        return

    print(f'Read audio file: {len(x_in)} samples, Sample Rate: {wav_fs} Hz')

    # --- Vectorized µ-law companding (transform linear to log domain) ---
    # xlog_in = sgn(x_in) * log(1 + mu * |x_in|) / log(1 + mu)
    log_1_plus_mu = np.log(1 + mu)
    xlog_in = np.sign(x_in) * np.log(1 + mu * np.abs(x_in)) / log_1_plus_mu

    # quantizer [-scale, scale]
    index = np.floor(xlog_in * scale) 

    # inverse-quantizer
    xlog_out = (index + 0.5) / scale

    # --- Vectorized µ-law expansion (transform log to linear domain) ---
    # x_out3 = sgn(xlog_out) * (1/mu) * ((1 + mu)**|xlog_out| - 1)
    x_out3 = np.sign(xlog_out) * (1 / mu) * (np.power(1 + mu, np.abs(xlog_out)) - 1)

    # write quantized wav file
    try:
        sf.write('x_out3.wav', x_out3, wav_fs, subtype='PCM_16')
        print("Wrote log-quantized audio to 'x_out3.wav'")
    except Exception as e:
        print(f"An error occurred while writing the audio file: {e}")

    # calc. SNR
    energy_sig = np.dot(x_in, x_in)
    x_diff = x_in - x_out3
    energy_noise = np.dot(x_diff, x_diff)

    if energy_noise == 0:
        snr_db = float('inf')
    else:
        snr_db = 10 * np.log10(energy_sig / energy_noise)

    print(f'Energy Signal = {energy_sig:.6f}')
    print(f'Energy Noise = {energy_noise:.6f}')
    print(f'SNR = {snr_db:.6f} [dB]')

    # plot waves
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax1.plot(x_in)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.set_title('Original Signal (x_in)')

    ax2.plot(x_out3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_title('Log-Quantized Signal (x_out3)')

    ax3.plot(x_diff)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True)
    ax3.set_title('Difference (x_diff)')

    plt.tight_layout()
    print("Showing plots...")

    # Play audio
    play_fs = 16000 # Hardcoded in original script
    try:
        print(f"\nPlaying original audio (at {play_fs} Hz)...")
        sd.play(x_in, play_fs)
        sd.wait()

        time.sleep(0.5)

        print(f"Playing log-quantized audio (at {play_fs} Hz)...")
        sd.play(x_out3, play_fs)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred during audio playback: {e}")

    print("--- End of log-quantizer script ---")
    plt.savefig('log_quantizer_plots.png')
    plt.show()

if __name__ == '__main__':
    run_log_quantizer()