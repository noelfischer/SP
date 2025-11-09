# qlloyd.py
# (Translation of qlloyd.m)

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time
from scipy.cluster.vq import kmeans, vq

def run_lloyd_quantizer():
    plt.close('all')
    print('--- Optimal quantizer design by Lloyd method ---')

    bit = 6
    num_levels = 2**bit
    print(f'Quantization bit = {bit} (Levels = {num_levels})')

    # read input wav file
    f_in = 'jikken16.wav'
    try:
        x_in, wav_fs = sf.read(f_in)
        info = sf.info(f_in)
        wav_nbit = 16 # Default
        if 'PCM_' in info.subtype:
            try:
                wav_nbit = int(info.subtype.split('_')[1])
            except:
                pass
    except FileNotFoundError:
        print(f"Error: '{f_in}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the audio file: {e}")
        return
        
    print(f'Read audio file: {len(x_in)} samples, Sample Rate: {wav_fs} Hz')

    # optimal quantizer design by Lloyd algorithm
    # MATLAB's lloyds(x_in, 2^bit) is 1D k-means.
    # We use scipy.cluster.vq.kmeans.
    # It requires data to be in shape (num_observations, num_features).
    # For 1D data, this is (N, 1).
    print("Running Lloyd's algorithm (k-means)...")
    x_in_1d = x_in.reshape(-1, 1)
    
    # codebook are the quantization levels (centroids)
    codebook, distortion = kmeans(x_in_1d, num_levels, iter=20)
    print("Lloyd's algorithm finished.")

    # Sort codebook for printing, as kmeans doesn't guarantee order
    codebook = np.sort(codebook, axis=0)
    
    print('Codebook (top 10 values):')
    print(codebook[:10].flatten())
    
    # Note: MATLAB's 'lloyds' also returns 'partition' (boundaries).
    # 'scipy.cluster.vq.kmeans' does not, but 'scipy.cluster.vq.vq'
    # (equivalent to MATLAB's 'quantiz') doesn't need it.

    # quantizer
    # 'vq' maps each observation in x_in_1d to the nearest codebook entry
    # 'index' contains the index of the nearest codebook entry for each sample
    index, distor_vq = vq(x_in_1d, codebook)

    # inverse-quantizer
    # Reconstruct the signal from the codebook
    x_out5 = codebook[index].flatten()

    # write quantized wav file
    try:
        # The original script does not specify bits, so we default to 16
        sf.write('x_out5.wav', x_out5, wav_fs, subtype='PCM_16')
        print("Wrote Lloyd-quantized audio to 'x_out5.wav'")
    except Exception as e:
        print(f"An error occurred while writing the audio file: {e}")

    # calc. SNR
    energy_sig = np.dot(x_in, x_in)
    x_diff = x_in - x_out5
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

    ax2.plot(x_out5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_title('Lloyd-Quantized Signal (x_out5)')

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

        print(f"Playing Lloyd-quantized audio (at {play_fs} Hz)...")
        sd.play(x_out5, play_fs)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred during audio playback: {e}")

    print("--- End of Lloyd-quantizer script ---")
    plt.savefig('lloyd_quantizer_plots.png')
    plt.show()

if __name__ == '__main__':
    run_lloyd_quantizer()