# Generate PNG figures for each vowel and create a LaTeX (.tex) file that includes them.
# Outputs:
#  - PNG images in /mnt/data/vowel_report_images/
#  - LaTeX file at /mnt/data/vowel_report.tex
#
# This uses the 8 kHz file /mnt/data/aeiou.wav. If missing, it will raise an error.

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, windows, freqz, lfilter
from matplotlib import rcParams
import os
rcParams.update({'figure.autolayout': True})

# Ensure directories
img_dir = "./vowel_report_images"
os.makedirs(img_dir, exist_ok=True)
tex_path = "./vowel_report.tex"

# Levinson-Durbin (same as before)
def levinson_durbin_lpc(y_signal, order):
    n = len(y_signal)
    R = np.zeros(order + 1)
    for i in range(order + 1):
        R[i] = np.sum(y_signal[:n-i] * y_signal[i:]) if n-i>0 else 0.0
    if R[0] <= 1e-12:
        a = np.zeros(order+1); a[0]=1.0; return a, 0.0
    a = np.zeros(order + 1); a[0] = 1.0
    err = R[0]
    for m in range(1, order+1):
        acc = R[m]
        for j in range(1, m):
            acc += a[j] * R[m-j]
        k = -acc / err if abs(err) > 1e-12 else 0.0
        a_prev = a.copy()
        a[m] = k
        for j in range(1, m):
            a[j] = a_prev[j] + k * a_prev[m-j]
        err = err * (1.0 - k*k)
        if err < 0: err = 0.0
    return a, err

def autocorr_pitch(x, fs, min_f0=60, max_f0=400):
    L = len(x)
    acf_full = np.correlate(x, x, mode='full')
    acf = acf_full[L-1:]
    if acf[0] == 0:
        return None, None, acf
    acf = acf / acf[0]
    min_lag = int(np.floor(fs / max_f0))
    max_lag = int(np.ceil(fs / min_f0))
    min_lag = max(1, min_lag)
    max_lag = min(len(acf)-1, max_lag)
    if max_lag <= min_lag:
        return None, None, acf
    search = acf[min_lag:max_lag+1]
    peak_idx = np.argmax(search)
    peak_lag = peak_idx + min_lag
    peak_val = acf[peak_lag]
    f0 = fs / peak_lag if peak_lag>0 else None
    return peak_lag, peak_val, acf

def pick_formants_from_lpc(lpc_db, freqs, num_formants=3, f_min=90, f_max=None):
    if f_max is None:
        f_max = freqs[-1]
    idx_min = np.searchsorted(freqs, f_min)
    idx_max = np.searchsorted(freqs, f_max)
    data = lpc_db[idx_min:idx_max]
    peaks, props = find_peaks(data, prominence=3)
    if len(peaks) == 0:
        return []
    peaks_global = peaks + idx_min
    peaks_sorted = np.sort(peaks_global)
    chosen = peaks_sorted[:num_formants]
    formants = [(freqs[i], lpc_db[i]) for i in chosen]
    return formants

# Read audio
wav8 = "./aeiou.wav"
if not os.path.exists(wav8):
    raise FileNotFoundError(f"Expected {wav8} to exist. Please upload the 8 kHz aeiou.wav file.")

data, fs = sf.read(wav8)
if data.ndim > 1:
    data = np.mean(data, axis=1)
data = data.astype(float)
if np.max(np.abs(data))>0:
    data = data / np.max(np.abs(data))

# Energy peak detection for vowels
win_len = int(0.010 * fs)
kernel = np.ones(win_len)/win_len
energy = np.convolve(data**2, kernel, mode='same')
separation = int(0.2 * fs)
peaks, props = find_peaks(energy, distance=separation, height=np.max(energy)*0.08)
if len(peaks) > 5:
    peak_heights = energy[peaks]
    top_idx = np.argsort(peak_heights)[-5:]
    peaks = np.sort(peaks[top_idx])
if len(peaks) < 5:
    peaks_all, _ = find_peaks(energy, distance=separation)
    peaks = peaks_all[:5]
if len(peaks) == 0:
    N=len(data); step=N//6; peaks=np.array([step*(i+1) for i in range(5)])
peaks = np.sort(peaks)[:5]

vowels = []
for i, center in enumerate(peaks):
    frame_len = int(0.025 * fs)
    start = max(0, center - frame_len//2)
    end = min(len(data), start + frame_len)
    frame = data[start:end]
    if len(frame) < frame_len:
        frame = np.pad(frame, (0, frame_len - len(frame)))
    win = windows.hamming(len(frame))
    win_frame = frame * win
    long_len = int(0.080 * fs)
    s_long = max(0, center - long_len//2)
    e_long = min(len(data), s_long + long_len)
    long_seg = data[s_long:e_long]
    if len(long_seg) < long_len:
        long_seg = np.pad(long_seg, (0, long_len - len(long_seg)))
    lag, peak_val, acf = autocorr_pitch(long_seg * windows.hamming(len(long_seg)), fs)
    f0 = fs/lag if lag is not None else None
    if lag is not None:
        n_display = int(5*lag)
        center_idx = len(long_seg)//2
        dstart = max(0, center_idx - n_display//2)
        dend = min(len(long_seg), dstart + n_display)
        display_seg = long_seg[dstart:dend]; display_time = np.arange(s_long + dstart, s_long + dend)/fs
    else:
        n_display = int(0.040*fs)
        dstart = max(0, frame_len//2 - n_display//2)
        dend = min(len(frame), dstart + n_display)
        display_seg = frame[dstart:dend]
        display_time = np.arange(start + dstart, start + dend)/fs

    fft_size = 4096
    Y = np.fft.rfft(win_frame, n=fft_size)
    Y_mag = np.abs(Y); freqs = np.fft.rfftfreq(fft_size, d=1/fs); spec_db = 20*np.log10(Y_mag + 1e-12)
    a, err = levinson_durbin_lpc(win_frame, 10)
    lpc_gain = np.sqrt(err) if err>0 else 1.0
    w_freqs, h = freqz([1.0], a, worN=fft_size//2+1, fs=fs)
    lpc_mag = np.abs(h) * lpc_gain; lpc_db = 20*np.log10(lpc_mag + 1e-12)
    formants = pick_formants_from_lpc(lpc_db, w_freqs, num_formants=3, f_min=90, f_max=fs/2)
    vowels.append({'index': i+1, 'center': int(center), 'start': int(start), 'end': int(end), 'f0': f0, 'lag': lag,
                   'display_time': display_time, 'display_seg': display_seg, 'long_seg': long_seg,
                   'freqs': freqs, 'spec_db': spec_db, 'lpc_freqs': w_freqs, 'lpc_db': lpc_db,
                   'lpc_coeffs': a, 'lpc_err': err, 'formants': formants})

# Save PNGs for each vowel: time waveform (5 periods), autocorr, FFT, LPC w/ formants, pole plot, residual+resynth
for v in vowels:
    idx = v['index']
    # Time waveform (~5 periods)
    plt.figure(figsize=(6.5,3))
    plt.plot(v['display_time'], v['display_seg'])
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
    title = f"Vowel {idx} - Time waveform (~5 pitch periods)"
    if v['f0'] is not None:
        title += f"  F0 ≈ {v['f0']:.1f} Hz"
    plt.title(title); plt.grid(True)
    fname = os.path.join(img_dir, f"vowel{idx}_time.png"); plt.savefig(fname, dpi=200); plt.close()

    # Autocorrelation (full) of long_seg
    plt.figure(figsize=(6.5,3))
    acf_full = np.correlate(v['long_seg']*windows.hamming(len(v['long_seg'])), v['long_seg']*windows.hamming(len(v['long_seg'])), mode='full')
    lags_full = np.arange(-len(v['long_seg'])+1, len(v['long_seg']))
    acf_norm = acf_full / (acf_full[len(acf_full)//2] + 1e-12)
    plt.plot(lags_full, acf_norm); plt.xlabel("Lag (samples)"); plt.ylabel("Normalized ACF"); plt.title(f"Vowel {idx} - Autocorrelation"); plt.grid(True)
    fname = os.path.join(img_dir, f"vowel{idx}_acf.png"); plt.savefig(fname, dpi=200); plt.close()

    # FFT spectrum (dB)
    plt.figure(figsize=(6.5,3))
    plt.plot(v['freqs'], v['spec_db']); plt.xlim(0, fs/2); plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)"); plt.title(f"Vowel {idx} - FFT (windowed frame)"); plt.grid(True)
    fname = os.path.join(img_dir, f"vowel{idx}_fft.png"); plt.savefig(fname, dpi=200); plt.close()

    # LPC spectrum + formants overlay
    plt.figure(figsize=(6.5,3))
    plt.plot(v['lpc_freqs'], v['lpc_db'], label='LPC envelope (dB)')
    plt.plot(v['freqs'], v['spec_db'], alpha=0.6, label='FFT (dB)')
    for j, fm in enumerate(v['formants']):
        plt.plot(fm[0], fm[1], 'ro'); plt.annotate(f"F{j+1}={fm[0]:.1f} Hz", xy=(fm[0], fm[1]), xytext=(fm[0]+80, fm[1]-10), arrowprops=dict(arrowstyle="->", lw=0.6))
    plt.xlim(0, fs/2); plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude (dB)"); plt.title(f"Vowel {idx} - LPC (1/A) and FFT"); plt.legend(); plt.grid(True)
    fname = os.path.join(img_dir, f"vowel{idx}_lpc.png"); plt.savefig(fname, dpi=200); plt.close()

    # Pole plot of A(z)
    from numpy import roots
    poles = roots(v['lpc_coeffs'])
    plt.figure(figsize=(4,4))
    ax = plt.gca()
    uc = plt.Circle((0,0),1, color='gray', fill=False, linestyle='--'); ax.add_patch(uc)
    plt.plot(np.real(poles), np.imag(poles), 'rx'); plt.title(f"Vowel {idx} - Poles of A(z)"); plt.xlabel("Real"); plt.ylabel("Imag"); plt.grid(True); plt.xlim([-1.1,1.1]); plt.ylim([-1.1,1.1])
    fname = os.path.join(img_dir, f"vowel{idx}_poles.png"); plt.savefig(fname, dpi=200); plt.close()

    # Residual and resynthesis (small example)
    res = lfilter(v['lpc_coeffs'], [1.0], v['display_seg'] if len(v['display_seg'])>0 else v['long_seg'][:len(v['display_seg'])])
    y2 = lfilter([1.0], v['lpc_coeffs'], res)
    plt.figure(figsize=(6.5,4))
    plt.subplot(3,1,1); plt.plot(v['display_time'], v['display_seg']); plt.title("Original segment (display)"); plt.grid(True)
    plt.subplot(3,1,2); plt.plot(np.arange(len(res)), res); plt.title("Residual"); plt.grid(True)
    plt.subplot(3,1,3); plt.plot(np.arange(len(y2)), y2); plt.title("Resynthesized (from residual)"); plt.grid(True)
    fname = os.path.join(img_dir, f"vowel{idx}_resynth.png"); plt.savefig(fname, dpi=200); plt.close()

# Also save an overall waveform + energy figure
plt.figure(figsize=(8,3))
t = np.arange(len(data))/fs
plt.plot(t, data); plt.xlabel("Time (s)"); plt.ylabel("Normalized amplitude"); plt.title("Full waveform (aeiou.wav)"); plt.grid(True)
for v in vowels:
    plt.axvline(v['center']/fs, color='r', linestyle='--')
    plt.text(v['center']/fs, 0.8, f"V{v['index']}", color='r')
fname = os.path.join(img_dir, "waveform_full.png"); plt.savefig(fname, dpi=200); plt.close()

plt.figure(figsize=(8,3)); plt.plot(np.arange(len(energy))/fs, energy); plt.xlabel("Time (s)"); plt.ylabel("Energy"); plt.title("Energy envelope (10 ms window)"); plt.grid(True)
fname = os.path.join(img_dir, "energy_env.png"); plt.savefig(fname, dpi=200); plt.close()

# Write LaTeX file
tex_lines = []
tex_lines.append(r"\documentclass[11pt,a4paper]{article}")
tex_lines.append(r"\usepackage{graphicx}")
tex_lines.append(r"\usepackage{caption}")
tex_lines.append(r"\usepackage{longtable}")
tex_lines.append(r"\usepackage{geometry}")
tex_lines.append(r"\geometry{margin=1in}")
tex_lines.append(r"\begin{document}")
tex_lines.append(r"\begin{center}\huge\textbf{Linear Prediction Analysis for five vowels (/a/, /i/, /u/, /e/, /o/)}\end{center}")
tex_lines.append(r"\vspace{1em}")
tex_lines.append(r"\section*{Data used}")
tex_lines.append(r"\textbf{File:} aeiou.wav (sample rate: %d Hz)\\ " % fs)
tex_lines.append(r"\includegraphics[width=\textwidth]{vowel_report_images/waveform_full.png}\\")
tex_lines.append(r"\includegraphics[width=\textwidth]{vowel_report_images/energy_env.png}\\")

tex_lines.append(r"\section*{1. Recording / Data}")
tex_lines.append(r"The audio file was inspected and five high-energy vowel segments were detected. The centers (sample indices) are listed below:")
tex_lines.append(r"\begin{verbatim}")
for v in vowels:
    tex_lines.append(f"Vowel {v['index']}: center={v['center']} samples  ({v['center']/fs:.3f} s)")
tex_lines.append(r"\end{verbatim}")

tex_lines.append(r"\section*{2. Time signals and pitch (F$_0$)}")
for v in vowels:
    tex_lines.append(r"\subsection*{Vowel %d}" % v['index'])
    tex_lines.append(r"\includegraphics[width=0.9\textwidth]{vowel_report_images/vowel%d_time.png}" % v['index'])
    tex_lines.append(r"\par Estimated F$_0$: %s Hz\\ " % (f"{v['f0']:.1f}" if v['f0'] is not None else "N/A"))

tex_lines.append(r"\section*{3. FFT power spectra (per vowel)}")
for v in vowels:
    tex_lines.append(r"\subsection*{Vowel %d}" % v['index'])
    tex_lines.append(r"\includegraphics[width=0.9\textwidth]{vowel_report_images/vowel%d_fft.png}" % v['index'])


tex_lines.append(r"\section*{4. 10-th order LPC parameters (frame length approx. 25 ms)}")
tex_lines.append(r"The LPC analysis used Levinson--Durbin to estimate a 10th-order AR model (coefficients $a[0]=1, a[1],...,a[10]$). The coefficients and prediction error power for each vowel are listed below.")
tex_lines.append(r"\usepackage{array}")  # ensure raggedright column available
tex_lines.append(r"\begin{longtable}{@{} l >{\raggedright\arraybackslash}p{10cm} @{} }")
tex_lines.append(r"\textbf{Vowel} & \textbf{LPC coefficients ($a[0]=1$) and error power}\\ \hline")

for v in vowels:
    coeffs = v['lpc_coeffs']
    # split long coefficient string into lines of about 5 values per line
    grouped = [coeffs[i:i+5] for i in range(0, len(coeffs), 5)]
    coeff_lines = " \\\\\n& ".join(
        [r"\verb|" + ", ".join([f"{c:.3e}" for c in group]) + r"|" for group in grouped]
    )
    tex_lines.append(
        rf"Vowel {v['index']} & Error power: {v['lpc_err']:.6g}\\"
        r"& LPC coeffs:\\"
        f"& {coeff_lines}\\\\[4pt]"
    )

tex_lines.append(r"\end{longtable}")

tex_lines.append(r"\section*{5. LPC power spectra and formants (visual observation)}")
tex_lines.append(r"Below are the LPC envelopes (1/A) plotted in dB, overlaid with the FFT for comparison. Formant estimates (F1, F2, F3) were obtained by observing the LPC envelope peaks.")
for v in vowels:
    tex_lines.append(r"\subsection*{Vowel %d}" % v['index'])
    tex_lines.append(r"\includegraphics[width=0.9\textwidth]{vowel_report_images/vowel%d_lpc.png}" % v['index'])
    if len(v['formants'])>0:
        formstr = ", ".join([f"{f[0]:.1f} Hz" for f in v['formants']])
    else:
        formstr = "N/A"
    tex_lines.append(r"\par Estimated formants: %s\\ " % formstr)

tex_lines.append(r"\section*{Appendix: additional program-style graphs}")
tex_lines.append(r"\subsection*{Autocorrelation examples and AR filter demo}")
tex_lines.append(r"\includegraphics[width=0.9\textwidth]{vowel_report_images/vowel1_acf.png}\\")
tex_lines.append(r"\includegraphics[width=0.9\textwidth]{vowel_report_images/vowel1_poles.png}\\")
tex_lines.append(r"\includegraphics[width=0.9\textwidth]{vowel_report_images/vowel1_resynth.png}\\")


tex_lines.append(r"\end{document}")

with open(tex_path, 'w') as f:
    f.write("\n".join(tex_lines))

print("Wrote LaTeX file to:", tex_path)
print("Wrote images to:", img_dir)
print("\nYou can download the .tex and the images folder. Compile the .tex with the images in the same relative path.")
