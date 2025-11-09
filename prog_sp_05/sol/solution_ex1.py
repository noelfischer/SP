# solution_ex1.py
#
# This script solves Exercise 1 by comparing three scalar quantizers
# for speech data at the same bit rate:
#   i) Linear Quantizer
#  ii) Non-linear (Logarithmic) Quantizer
# iii) Optimal Quantizer (Lloyd's Algorithm)
#
# We will load a single speech file and process it with each method,
# then compare the resulting Signal-to-Noise Ratio (SNR).
# A higher SNR means the quantized signal is of higher quality
# (less noise/error).

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from scipy.cluster.vq import kmeans, vq

# --- Configuration ---
# Set the bit rate to be used FOR ALL THREE quantizers.
# This allows for a fair comparison.
BIT_RATE = 6
INPUT_FILE = 'jikken16.wav'

# Set a scale for linear quantizer. Log and Lloyd are adaptive.
LINEAR_SCALE = 2**(BIT_RATE - 1)

# Set Mu for logarithmic quantizer
MU = 255.0

# --- Helper Function for SNR ---
def calculate_snr(original, quantized):
    """Calculates the Signal-to-Noise Ratio (SNR) in dB."""
    energy_sig = np.dot(original, original)
    x_diff = original - quantized
    energy_noise = np.dot(x_diff, x_diff)
    
    if energy_noise == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(energy_sig / energy_noise)
    return snr_db

def run_exercise_1():
    plt.close('all')
    print(f"--- Running Quantizer Comparison (All at {BIT_RATE} bits) ---")

    # --- Load Audio Data ---
    try:
        x_in, wav_fs = sf.read(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: '{INPUT_FILE}' not found.")
        print("Please place 'jikken16.wav' in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred reading the audio file: {e}")
        return

    print(f"Loaded '{INPUT_FILE}': {len(x_in)} samples, {wav_fs} Hz")

    # --- i) Linear Quantizer ---
    print("\nProcessing Linear Quantizer...")
    # Quantizer: [ -scale, scale ]
    index_lin = np.floor(x_in * LINEAR_SCALE)
    # Inverse-quantizer: (index + 0.5) / scale
    x_out_lin = (index_lin + 0.5) / LINEAR_SCALE
    snr_lin = calculate_snr(x_in, x_out_lin)
    
    # --- ii) Non-linear (Logarithmic) Quantizer ---
    print("Processing Logarithmic (µ-law) Quantizer...")
    scale_log = 2**(BIT_RATE - 1)
    
    # Companding (linear to log domain)
    log_1_plus_mu = np.log(1 + MU)
    xlog_in = np.sign(x_in) * np.log(1 + MU * np.abs(x_in)) / log_1_plus_mu
    
    # Quantizer (in log domain)
    index_log = np.floor(xlog_in * scale_log)
    # Inverse-quantizer (in log domain)
    xlog_out = (index_log + 0.5) / scale_log
    
    # Expansion (log to linear domain)
    x_out_log = np.sign(xlog_out) * (1 / MU) * (np.power(1 + MU, np.abs(xlog_out)) - 1)
    snr_log = calculate_snr(x_in, x_out_log)

    # --- iii) Optimal Quantizer (Lloyd's Algorithm) ---
    print("Processing Optimal (Lloyd) Quantizer...")
    num_levels = 2**BIT_RATE
    
    # Reshape for 1D k-means
    x_in_1d = x_in.flatten()
    
    # Run Lloyd's algorithm (k-means) to find the codebook (centroids)
    # This finds the 'optimal' quantization levels based on the signal's stats
    codebook_lloyd, _ = kmeans(x_in_1d, num_levels, iter=40)
    
    # 'vq' (Vector Quantize) finds the nearest codebook entry for each sample
    index_lloyd, _ = vq(x_in_1d, codebook_lloyd)
    
    # Reconstruct the signal from the codebook
    x_out_lloyd = codebook_lloyd[index_lloyd].flatten()
    snr_lloyd = calculate_snr(x_in, x_out_lloyd)

    # --- Performance Explanation & Comparison ---
    print("\n--- Performance Comparison ---")
    print(f"Bit Rate: {BIT_RATE} bits (for all methods)")
    print(f"  i) Linear Quantizer SNR:     {snr_lin:9.4f} dB")
    print(f" ii) Logarithmic Quantizer SNR: {snr_log:9.4f} dB")
    print(f"iii) Optimal (Lloyd) SNR:      {snr_lloyd:9.4f} dB")

    print("\n--- Explanation ---")
    print("Linear Quantizer:")
    print("  - Uses uniform step sizes. This is inefficient for speech,")
    print("    which has many low-amplitude values and few high-amplitude ones.")
    print("  - It wastes quantization levels on 'empty' high-amplitude regions,")
    print("    resulting in the lowest SNR (most noise).")
    print("\nLogarithmic (µ-law) Quantizer:")
    print("  - Uses non-uniform step sizes: fine steps for low amplitudes")
    print("    and coarse steps for high amplitudes.")
    print("  - This matches the probability distribution of speech signals well,")
    print("    leading to a significant SNR improvement over the linear quantizer.")
    print("\nOptimal (Lloyd) Quantizer:")
    print("  - Uses k-means to find the 'best' possible quantization levels")
    print("    (codebook) *specifically for this audio file*.")
    print("  - It adapts to the precise distribution of the input signal.")
    print("  - As expected, it provides the highest SNR, as it is")
    print("    mathematically 'optimal' for this specific data.")

    # --- Plotting ---
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True, sharey=True)
    
    fig.suptitle(f"Quantizer Comparison at {BIT_RATE} bits", fontsize=16)

    axes[0].plot(x_in, label='Original Signal')
    axes[0].set_title("Original Signal (x_in)")
    axes[0].legend(loc='upper right')

    axes[1].plot(x_out_lin, 'r', label=f"Linear (SNR: {snr_lin:.2f} dB)")
    axes[1].set_title("i) Linear Quantizer")
    axes[1].legend(loc='upper right')

    axes[2].plot(x_out_log, 'g', label=f"Logarithmic (SNR: {snr_log:.2f} dB)")
    axes[2].set_title("ii) Logarithmic (µ-law) Quantizer")
    axes[2].legend(loc='upper right')
    axes[2].set_ylabel("Amplitude")

    axes[3].plot(x_out_lloyd, 'm', label=f"Optimal (SNR: {snr_lloyd:.2f} dB)")
    axes[3].set_title("iii) Optimal (Lloyd) Quantizer")
    axes[3].legend(loc='upper right')
    axes[3].set_xlabel("Sample Number")

    for ax in axes:
        ax.grid(True)
        
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    print("\nShowing plots...")
    plt.savefig('quantizer_comparison.png')

    # --- Audio Playback ---
    play_fs = 16000  # Use a consistent playback rate
    try:
        print(f"\nPlaying original audio (at {play_fs} Hz)...")
        sd.play(x_in, play_fs)
        sd.wait()
        time.sleep(0.5)

        print(f"Playing Linear quantized audio...")
        sd.play(x_out_lin, play_fs)
        sd.wait()
        time.sleep(0.5)

        print(f"Playing Logarithmic quantized audio...")
        sd.play(x_out_log, play_fs)
        sd.wait()
        time.sleep(0.5)

        print(f"Playing Optimal (Lloyd) quantized audio...")
        sd.play(x_out_lloyd, play_fs)
        sd.wait()
        
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred during audio playback: {e}")

    print("\n--- End of Exercise 1 ---")
    #plt.waitforbuttonpress() # Wait for user to close plot

if __name__ == '__main__':
    run_exercise_1()