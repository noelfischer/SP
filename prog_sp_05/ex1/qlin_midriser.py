# qlin_midriser.py
# (Translation of qlin_midriser.m)

import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import time

def run_linear_quantizer():
    plt.close('all')
    print('--- Linear quantization, midriser type ---')

    bit = 7
    scale = 2**(bit - 1)
    print(f'Quantization bit = {bit}')
    print(f'Scale = {scale}')

    # read input wav file
    file_in = 'jikken16.wav'
    try:
        x_in, wav_fs = sf.read(file_in)
        info = sf.info(file_in)
        # Attempt to get bit depth from subtype, e.g., 'PCM_16' -> 16
        wav_nbit = 16 # Default
        if 'PCM_' in info.subtype:
            try:
                wav_nbit = int(info.subtype.split('_')[1])
            except:
                pass # Keep default
    except FileNotFoundError:
        print(f"Error: '{file_in}' not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading the audio file: {e}")
        return

    print(f'Input file = {file_in}')
    print(f'Sample Rate (wav_fs) = {wav_fs} [Hz]')
    print(f'Bit Depth (wav_nbit) = {wav_nbit} [bit]')

    # quantizer [-scale, scale]
    index = np.floor(x_in * scale) 

    # inverse-quantizer
    x_out = (index + 0.5) / scale

    # write quantized wav file
    file_out = 'x_out.wav'
    subtype_out = f'PCM_{wav_nbit}' if wav_nbit in [16, 24, 32] else 'PCM_16'
    
    try:
        sf.write(file_out, x_out, wav_fs, subtype=subtype_out)
        print(f"Wrote quantized audio to '{file_out}', Fs = {wav_fs} [Hz], bits = {wav_nbit} [bit]")
    except Exception as e:
        print(f"An error occurred while writing the audio file: {e}")

    # calc. SNR
    energy_sig = np.dot(x_in, x_in)
    x_diff = x_in - x_out
    energy_noise = np.dot(x_diff, x_diff)
    
    if energy_noise == 0:
        snr_db = float('inf') # Avoid division by zero
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

    ax2.plot(x_out)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_title('Quantized Signal (x_out)')

    ax3.plot(x_diff)
    ax3.set_xlabel('Time')
    ax3.set_ylabel('Amplitude')
    ax3.grid(True)
    ax3.set_title('Difference (x_diff)')

    plt.tight_layout()
    print("Showing plots...")
    plt.show(block=False) # Show plot but don't block audio

    # Play audio
    # Note: Original script hardcodes 16000 Hz for playback.
    play_fs = 16000
    try:
        print(f"\nPlaying original audio (at {play_fs} Hz)...")
        sd.play(x_in, play_fs)
        sd.wait() # Wait for playback to finish

        time.sleep(0.5) # Brief pause

        print(f"Playing quantized audio (at {play_fs} Hz)...")
        sd.play(x_out, play_fs)
        sd.wait()
        print("Playback finished.")
    except Exception as e:
        print(f"An error occurred during audio playback: {e}")
        print("You may need to configure your audio device.")
        
    print("--- End of linear quantizer script ---")
    plt.waitforbuttonpress() # Wait for user to close plot

if __name__ == '__main__':
    run_linear_quantizer()