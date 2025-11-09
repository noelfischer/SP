# sample.py
# (Translation of sample.m)

import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

def run_sample():
    # read wav file
    try:
        x16, fs = sf.read('jikken16.wav')
    except FileNotFoundError:
        print("Error: 'jikken16.wav' not found.")
        print("Please place the audio file in the same directory as the script.")
        return
    except Exception as e:
        print(f"An error occurred while reading the audio file: {e}")
        return

    print(f"Read audio file: {len(x16)} samples, Sample Rate: {fs} Hz")

    # write the processed wav file 
    x16_scale = x16 * 1.5
    
    # Clip values to be within the valid range [-1.0, 1.0] for audio
    x16_scale = np.clip(x16_scale, -1.0, 1.0)

    # Note: The original MATLAB script hardcodes 16000 Hz for writing.
    # It also uses an old syntax 'audiowrite(..., 16)' for 16-bit.
    # The Python equivalent is subtype='PCM_16'.
    # We will use the *read* sample rate 'fs' for correctness, 
    # but the hardcoded 16-bit part is preserved.
    out_fs = 16000 # As in original script
    try:
        sf.write('x16_scale.wav', x16_scale, out_fs, subtype='PCM_16')
        print(f"Wrote scaled audio to 'x16_scale.wav' at {out_fs} Hz")
    except Exception as e:
        print(f"An error occurred while writing the audio file: {e}")

    # plot waves
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))
    
    ax1.plot(x16)
    ax1.set_xlabel('x16 Time')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True)
    ax1.set_title('Original Audio (x16)')

    ax2.plot(x16_scale)
    ax2.set_xlabel('x16scale Time')
    ax2.set_ylabel('Amplitude')
    ax2.grid(True)
    ax2.set_title('Scaled Audio (x16_scale)')

    plt.tight_layout()
    print("Showing plots...")
    plt.show()

if __name__ == '__main__':
    run_sample()