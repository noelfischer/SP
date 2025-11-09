import soundfile as sf
import numpy as np
from scipy.cluster.vq import kmeans, vq
import matplotlib.pyplot as plt

def run_lloyd_quantizer():

    print('--- Lloyd quantizer: SNR focused version ---')

    bit = 6
    num_levels = 2**bit
    print(f'Using {bit} bits with {num_levels} levels')

    f_in = 'jikken16.wav'
    x_in, wav_fs = sf.read(f_in)

    # Flatten so the algorithm sees a true 1D signal
    x_flat = x_in.flatten()

    # Normalize to improve centroid stability
    peak = np.max(np.abs(x_flat))
    if peak == 0:
        print('Input is silent')
        return
    x_norm = x_flat / peak

    print('Running Lloyd iterations')

    # Use more iterations than default to seek better convergence
    codebook, distortion = kmeans(x_norm, num_levels, iter=50)

    # Sort centroids for stable indexing
    codebook = np.sort(codebook)

    # Quantize
    idx, _ = vq(x_norm, codebook)
    x_q = codebook[idx]

    # Denormalize
    x_q = x_q * peak

    # Save output
    sf.write('x_lloyd_snr.wav', x_q, wav_fs, subtype='PCM_16')
    print('Saved quantized file')

    # SNR
    noise = x_flat - x_q
    e_signal = np.dot(x_flat, x_flat)
    e_noise = np.dot(noise, noise)

    if e_noise == 0:
        snr_db = float('inf')
    else:
        snr_db = 10 * np.log10(e_signal / e_noise)

    print(f'SNR: {snr_db:.4f} dB')

    # Plots
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(x_flat)
    plt.title('Original')

    plt.subplot(3, 1, 2)
    plt.plot(x_q)
    plt.title('Quantized')

    plt.subplot(3, 1, 3)
    plt.plot(noise)
    plt.title('Noise')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    run_lloyd_quantizer()
