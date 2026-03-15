"""
Part 2: HRV analysis from 180s PPG recording.

Uses a zero-phase Butterworth bandpass (0.5–5 Hz) via filtfilt to cleanly
isolate pulse content, then does time-domain peak detection and reports
HRV metrics in milliseconds.
"""
from scipy.signal import find_peaks, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

# ── Load data ──
data = np.loadtxt("ppg_data_180s_hrv.csv", delimiter=",", skiprows=1)
t = data[:, 0]
ppg_raw = data[:, 1]
N = len(ppg_raw)
actual_fs = (N - 1) / (t[-1] - t[0])
print(f"N = {N}, actual fs = {actual_fs:.2f} Hz")

# ── Bandpass filter: 0.5–5 Hz, 4th-order Butterworth, zero-phase ──
BP_LO, BP_HI = 0.65, 3
ORDER = 4
b, a = butter(ORDER, [BP_LO, BP_HI], btype='band', fs=actual_fs)
ppg_filtered = filtfilt(b, a, ppg_raw)

# ── Figure 1: Raw vs. filtered time domain ──
fig1, (ax_raw, ax_filt) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

ax_raw.plot(t, ppg_raw, linewidth=0.8)
ax_raw.set_ylabel("Voltage (V)")
ax_raw.set_title("Original PPG – Time Domain")
ax_raw.grid(True, alpha=0.3)

ax_filt.plot(t, ppg_filtered, linewidth=0.8, color='tab:orange')
ax_filt.set_xlabel("Time (s)")
ax_filt.set_ylabel("Voltage (V)")
ax_filt.set_title(f"Bandpass Filtered PPG ({BP_LO}–{BP_HI} Hz, order {ORDER}, zero-phase)")
ax_filt.grid(True, alpha=0.3)

fig1.tight_layout()
plt.show()

# ── Figure 2: FFT before and after filtering ──
dt = 1 / actual_fs
freq = np.fft.fftfreq(N, dt)
freq_pos = freq[1:N//2]

mag_raw = np.abs(np.fft.fft(ppg_raw - np.mean(ppg_raw))) / N
mag_raw_pos = mag_raw[1:N//2]

mag_filt = np.abs(np.fft.fft(ppg_filtered)) / N
mag_filt_pos = mag_filt[1:N//2]

filt_fund_idx = 1 + np.argmax(mag_filt_pos)
filt_fund_freq = freq[filt_fund_idx]
filt_fund_mag = mag_filt[filt_fund_idx]
print(f"Filtered FFT fundamental: {filt_fund_freq:.4f} Hz --> {filt_fund_freq * 60:.1f} BPM")

fig2, (ax_fft_raw, ax_fft_filt) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

ax_fft_raw.stem(freq_pos, mag_raw_pos)
ax_fft_raw.set_ylabel("Magnitude")
ax_fft_raw.set_title("FFT – Raw Signal (DC removed)")
ax_fft_raw.set_xlim(0, 10)
ax_fft_raw.grid(True, alpha=0.3)

ax_fft_filt.stem(freq_pos, mag_filt_pos)
ax_fft_filt.scatter(filt_fund_freq, filt_fund_mag, color='red', s=60, zorder=5,
                    label=f'Fundamental ({filt_fund_freq:.4f} Hz / {filt_fund_freq*60:.1f} BPM)')
ax_fft_filt.set_xlabel("Frequency [Hz]")
ax_fft_filt.set_ylabel("Magnitude")
ax_fft_filt.set_title(f"FFT – After Bandpass ({BP_LO}–{BP_HI} Hz)")
ax_fft_filt.set_xlim(0, 10)
ax_fft_filt.legend()
ax_fft_filt.grid(True, alpha=0.3)

fig2.tight_layout()
plt.show()

# ── Peak detection on filtered signal ──
min_prominence = 0.1 * (ppg_filtered.max() - ppg_filtered.min())
peak_indices, _ = find_peaks(ppg_filtered, distance=int(0.4 * actual_fs), prominence=min_prominence)
peak_times = t[peak_indices]
ibi_s = np.diff(peak_times)
ibi_ms = ibi_s * 1000.0

print(f"\n── HRV Report ──")
print(f"Peaks detected: {len(peak_indices)}")
print(f"Mean IBI:   {np.mean(ibi_ms):.1f} ms")
print(f"SDNN:       {np.std(ibi_ms):.1f} ms")
print(f"RMSSD:      {np.sqrt(np.mean(np.diff(ibi_ms)**2)):.1f} ms")
print(f"IBI range:  {ibi_ms.min():.1f} – {ibi_ms.max():.1f} ms")
print(f"Mean HR:    {60000.0 / np.mean(ibi_ms):.1f} BPM")
print(f"HR range:   {60000.0 / ibi_ms.max():.1f} – {60000.0 / ibi_ms.min():.1f} BPM")

# ── Figure 3: Filtered signal with detected peaks ──
fig3, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, ppg_filtered, linewidth=0.8, color='tab:orange')
ax.plot(peak_times, ppg_filtered[peak_indices], 'ro', markersize=4, label='Peaks')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_title(f"Filtered PPG + Peak Detection")
ax.legend()
ax.set_xlim(0,50)
ax.grid(True, alpha=0.3)
fig3.tight_layout()
plt.show()

