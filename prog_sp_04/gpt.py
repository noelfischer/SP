# Retry: run analysis again with error capture and print messages.
import numpy as np, os, matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import correlate, lfilter
plt.rcParams['figure.dpi'] = 150
def lpc_autocorr(x, order):
    r = np.array([np.sum(x[:len(x)-k] * x[k:]) for k in range(order+1)], dtype=float)
    a = np.zeros(order+1)
    e = r[0]
    a[0]=1.0
    if e == 0:
        return a, e
    for i in range(1, order+1):
        acc = r[i]
        for j in range(1,i):
            acc += a[j]*r[i-j]
        k = -acc/e
        a_temp = a.copy()
        a[i]=k
        for j in range(1,i):
            a[j] = a_temp[j] + k * a_temp[i-j]
        e = e * (1 - k*k)
        if e <= 0:
            e = 1e-8
    return a,e

frame_ms=30; shift_ms=5; order=10
files=['./aiueo3_08k.wav','./M01_mip_08k.wav', './F01_mip_08k.wav']
out_dir='./analysis_outputs'
os.makedirs(out_dir, exist_ok=True)
results={}
for filepath in files:
    try:
        Fs,x = wavfile.read(filepath)
    except Exception as e:
        print("Error reading", filepath, ":", e)
        continue
    if x.ndim>1:
        x = x.mean(axis=1)
    x = x.astype(float)
    N_frame = int(round(frame_ms/1000 * Fs))
    N_shift = int(round(shift_ms/1000 * Fs))
    window = np.hamming(N_frame)
    num_frames = max(0, (len(x)-N_frame)//N_shift + 1)
    pitches = np.zeros(num_frames); voiced_flags = np.zeros(num_frames, dtype=bool)
    lpc_coefs = np.zeros((num_frames, order+1)); lpc_sigma = np.zeros(num_frames); times=np.zeros(num_frames)
    min_f0=50.0; max_f0=400.0
    min_lag = int(Fs/max_f0); max_lag=int(Fs/min_f0)
    for i in range(num_frames):
        start = i * N_shift
        frame = x[start:start+N_frame] * window
        times[i] = (start + N_frame/2)/Fs
        r = correlate(frame, frame, mode='full')
        mid = len(r)//2
        rpos = r[mid + min_lag: mid + max_lag + 1]
        if len(rpos)==0:
            voiced_flags[i]=False; pitches[i]=0.0
        else:
            k_rel = np.argmax(rpos)
            k = k_rel + min_lag
            Rmax = rpos[k_rel] / (r[mid]+1e-12)
            energy = np.sum(frame**2)
            if Rmax > 0.35 and energy > 1e-6:
                voiced_flags[i]=True; pitches[i]=Fs/k
            else:
                voiced_flags[i]=False; pitches[i]=0.0
        a,e = lpc_autocorr(frame, order)
        lpc_coefs[i,:]=a; lpc_sigma[i]=e
    # save pitch plot
    import matplotlib
    plt.figure(figsize=(8,2))
    plt.plot(times, pitches, '-o', markersize=2)
    plt.title(os.path.basename(filepath) + ' - Pitch contour (autocorr)')
    plt.xlabel('Time (s)'); plt.ylabel('Freq (Hz)')
    plt.ylim(0,450); plt.grid(True)
    pitch_png = os.path.join(out_dir, os.path.basename(filepath).replace('.wav','_pitch.png'))
    plt.tight_layout(); plt.savefig(pitch_png); plt.close()
    # spectral envelopes
    dft=1024; freqs = np.linspace(0, Fs/2, dft//2+1)
    picks = [max(0, num_frames//6), max(0, num_frames//3), max(0, num_frames//2), max(0, num_frames-5)]
    plt.figure(figsize=(8,4))
    for idx in picks:
        a = lpc_coefs[idx,:]; sigma = lpc_sigma[idx]
        A = np.fft.fft(a, dft)[:dft//2+1]
        P = sigma / (np.abs(A)**2 + 1e-12)
        plt.plot(freqs, 10*np.log10(P+1e-12), label=f'frame {idx} t={times[idx]:.2f}s V={voiced_flags[idx]}')
    plt.title(os.path.basename(filepath) + ' - LPC spectral envelopes (order 10)')
    plt.xlabel('Hz'); plt.ylabel('Power (dB)'); plt.legend(fontsize=6); plt.grid(True)
    spec_png = os.path.join(out_dir, os.path.basename(filepath).replace('.wav','_lpc_env.png'))
    plt.tight_layout(); plt.savefig(spec_png); plt.close()
    # residual acf for a voiced frame
    res_ac_png = None
    voiced_idx = np.where(voiced_flags)[0]
    if voiced_idx.size>0:
        idx = voiced_idx[len(voiced_idx)//2]
        start = int(idx * N_shift)
        frame = x[start:start+N_frame] * window
        a = lpc_coefs[idx,:]
        residual = lfilter(a, 1.0, frame)
        rres = correlate(residual, residual, mode='full')
        lags = np.arange(-len(residual)+1, len(residual))
        plt.figure(figsize=(6,3))
        mid = len(rres)//2; span = min(200, len(rres)//2)
        plt.plot(lags[mid-span:mid+span+1], rres[mid-span:mid+span+1])
        plt.title(os.path.basename(filepath) + f' - Residual autocorr frame {idx} t={times[idx]:.2f}s')
        plt.xlabel('Lag (samples)'); plt.grid(True)
        res_ac_png = os.path.join(out_dir, os.path.basename(filepath).replace('.wav',f'_res_acf_frame{idx}.png'))
        plt.tight_layout(); plt.savefig(res_ac_png); plt.close()
    # vocoder synthesis
    synth = np.zeros_like(x)
    for i in range(num_frames):
        start = i * N_shift; frame_len = N_frame
        a = lpc_coefs[i,:]; sigma = lpc_sigma[i]
        if voiced_flags[i] and pitches[i]>0:
            period = int(round(Fs / pitches[i])); 
            if period <=0: period = int(Fs/100)
            frame_exc = np.zeros(frame_len)
            offset = frame_len//2
            for ppos in range(offset % period, frame_len, period):
                frame_exc[ppos] = np.sqrt(sigma)
        else:
            frame_exc = np.random.randn(frame_len) * np.sqrt(sigma+1e-12)
        frame_synth = lfilter([1.0], a, frame_exc) * window
        synth[start:start+frame_len] += frame_synth
    synth = synth / (np.max(np.abs(synth))+1e-9) * 0.99
    synth_wav = os.path.join(out_dir, os.path.basename(filepath).replace('.wav','_synth.wav'))
    synth = synth * 32767
    wavfile.write(synth_wav, Fs, (synth).astype(np.int16))
    # waveform plot
    t = np.arange(len(x))/Fs
    plt.figure(figsize=(8,3))
    seg_end = min(len(x), Fs*1)
    plt.plot(t[:seg_end], x[:seg_end], label='original', alpha=0.7)
    plt.plot(t[:seg_end], synth[:seg_end], label='synth', alpha=0.7)
    plt.title(os.path.basename(filepath) + ' - original vs synthesized (first 1s)')
    plt.xlabel('Time (s)'); plt.legend(fontsize=6); plt.grid(True)
    wave_png = os.path.join(out_dir, os.path.basename(filepath).replace('.wav','_orig_synth.png'))
    plt.tight_layout(); plt.savefig(wave_png); plt.close()
    results[os.path.basename(filepath)] = {'Fs':Fs,'num_frames':num_frames,'pitch_png':pitch_png,'spec_png':spec_png,'res_ac_png':res_ac_png,'synth_wav':synth_wav,'wave_png':wave_png}
print("Done. Outputs in", out_dir)
results

