# prog_sp_02-1.py�@// English ver.
import wave
import numpy as np
import matplotlib.pyplot as plt
import sys
import IPython.display as display

def analyze_wav(file_path):
    """
    Function to read a WAV file and display/plot its header information and waveform data.

    Args:
        file_path (str): Path to the WAV file
    """
    try:
        # Open the WAV file
        with wave.open(file_path, 'rb') as wf:
            # Get header information
            nchannels = wf.getnchannels()        # Number of channels
            sampwidth = wf.getsampwidth()        # Sample width (bytes)
            framerate = wf.getframerate()        # Sampling frequency
            nframes = wf.getnframes()            # Number of frames
            comptype = wf.getcomptype()          # Compression type (should be 'NONE')
            compname = wf.getcompname()          # Compression name (should be 'not compressed')

            # Display header information
            print("--- WAV Header Information ---")
            print(f"File Name: {file_path}")
            print(f"Number of Channels: {nchannels}")
            print(f"Sample Width (bytes): {sampwidth}")
            print(f"Sampling Frequency (Hz): {framerate}")
            print(f"Number of Frames: {nframes}")
            print(f"Compression Type: {comptype}")
            print(f"Compression Name: {compname}")
            print("-----------------------------")

            # Read the audio data
            # readframes returns a bytes object
            frames = wf.readframes(nframes)

            # Convert byte data to a numpy array
            # The data type depends on the sample width
            if sampwidth == 1:
                # 8 bit unsigned
                dtype = np.uint8
            elif sampwidth == 2:
                # 16 bit signed
                dtype = np.int16
            elif sampwidth == 3:
                 # 24 bit signed (often treated as int32 in numpy)
                 # The bytes read by readframes must be converted to an appropriate form
                 # For simplicity here, we show an example of reading 2 bytes at a time and treating it as int16,
                 # but this cannot be treated as strict 24-bit data.
                 # Separate conversion processing is required to accurately handle 24-bit data.
                 print("Warning: 24bit WAV found. Treating as 16bit for simplicity. Accuracy may be affected.")
                 dtype = np.int16
                 # Example of accurately handling 24bit (commented out here)
                 # audio_data = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3)
                 # audio_data = (audio_data[:, 2].astype(np.int32) << 16) | (audio_data[:, 1].astype(np.int32) << 8) | audio_data[:, 0].astype(np.int32)
                 # audio_data[audio_data >= 0x800000] -= 0x1000000
                 # audio_data = audio_data.astype(np.float32) / (2**23) # For normalization
            elif sampwidth == 4:
                # 32 bit signed or float
                # Usually int32 if comptype='NONE'
                dtype = np.int32
            else:
                raise ValueError(f"Unsupported sample width: {sampwidth} bytes")

            # Convert the byte sequence to a numerical array using numpy.frombuffer
            # dtype is the data type, count is the number of elements to read (here, all frames * number of channels)
            # offset is the starting position for reading (here, 0)
            audio_data = np.frombuffer(frames, dtype=dtype)

            # Split data by channel (for stereo)
            if nchannels > 1:
                # Reshape so each row is one frame and each column is a channel
                audio_data = audio_data.reshape(-1, nchannels)
 
            # Create time axis
            duration = nframes / framerate
            time = np.linspace(0., duration, nframes)

            # Plot the waveform
            plt.figure(figsize=(12, 6))

            if nchannels == 1:
                # For mono
                plt.plot(time, audio_data)
                plt.title('Waveform (Mono)')
                plt.ylabel('Amplitude')
            else:
                # For stereo or higher, plot each channel
                for i in range(nchannels):
                    plt.plot(time, audio_data[:, i], label=f'Channel {i+1}')
                plt.title('Waveform (Stereo)')
                plt.ylabel('Amplitude')
                plt.legend()

            plt.xlabel('Time [s]')
            plt.grid(True)
            plt.show()

            return audio_data, framerate

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except wave.Error as e:
        print(f"Error processing WAV file: {e}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    file_path = "./aiueo2_16k_work.wav"
    audio_data, fs = analyze_wav(file_path)

    print('audio_data.dtype', audio_data.dtype)
    print('fs=',fs)

    y = audio_data/32768 #divide 32768 to convert to [-1,1] floating values.

    #plt.plot(y)

    print('y.dtype, len(y), max(y), min(y)', y.dtype, len(y), max(y), min(y))

    #explicit representation for IPython.display.Audio(y,rate=fs), as the code is in the if clause.
    display.display(
        display.Audio(y, rate=fs)
    )
