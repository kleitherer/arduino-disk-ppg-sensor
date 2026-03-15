"""
Part 2: HRV analysis from 180s PPG recording.

Uses a zero-phase Butterworth bandpass (0.5–5 Hz) via filtfilt to cleanly
isolate pulse content, then does time-domain peak detection and reports
HRV metrics in milliseconds.
"""
from scipy.signal import find_peaks, butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("ppg_data_180s_hrv.csv", delimiter=",", skiprows=1)
t_full = data[:, 0]
ppg_full = data[:, 1]
actual_fs = (len(ppg_full) - 1) / (t_full[-1] - t_full[0])

TRIM_START = 10
TRIM_END = 20
keep = (t_full >= t_full[0] + TRIM_START) & (t_full <= t_full[-1] - TRIM_END)
t = t_full[keep]
ppg_raw = ppg_full[keep]
N = len(ppg_raw)
print(f"N = {N}, actual fs = {actual_fs:.2f} Hz  (trimmed first {TRIM_START}s, last {TRIM_END}s)")

# ── Two-stage filter: highpass then lowpass, 4th-order Butterworth, zero-phase ──
HP_CUTOFF = 0.4
LP_CUTOFF = 6.5
ORDER = 4

b_hp, a_hp = butter(ORDER, HP_CUTOFF, btype='high', fs=actual_fs)
ppg_hp = filtfilt(b_hp, a_hp, ppg_raw)

b_lp, a_lp = butter(ORDER, LP_CUTOFF, btype='low', fs=actual_fs)
ppg_filtered = filtfilt(b_lp, a_lp, ppg_hp)

# ── Figure 1: Raw vs. filtered time domain ──
fig1, (ax_raw, ax_filt) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

ax_raw.plot(t, ppg_raw, linewidth=0.8)
ax_raw.set_ylabel("Voltage (V)")
ax_raw.set_title("Original PPG – Time Domain")
ax_raw.grid(True, alpha=0.3)
ax_raw.set_xlim(10,40)

ax_filt.plot(t, ppg_filtered, linewidth=0.8, color='tab:orange')
ax_filt.set_xlabel("Time (s)")
ax_filt.set_ylabel("Voltage (V)")
ax_filt.set_title(f"Filtered PPG (HP {HP_CUTOFF} Hz + LP {LP_CUTOFF} Hz, order {ORDER}, zero-phase)")
ax_filt.grid(True, alpha=0.3)
ax_filt.set_xlim(10,40)


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
ax_fft_filt.set_title(f"FFT – After HP {HP_CUTOFF} Hz + LP {LP_CUTOFF} Hz")
ax_fft_filt.set_xlim(0, 10)
ax_fft_filt.legend()
ax_fft_filt.grid(True, alpha=0.3)

fig2.tight_layout()
plt.show()

# ── Peak detection on filtered signal ──
min_prominence = 0.04 * (ppg_filtered.max() - ppg_filtered.min())
MAX_PEAK_HEIGHT = 0.13
MIN_PEAK_HEIGHT = 0.01
peak_indices, _ = find_peaks(
    ppg_filtered, 
    distance=int(0.52 * actual_fs), 
    width=int(0.08 * actual_fs), 
    prominence=min_prominence, 
    height=(MIN_PEAK_HEIGHT, MAX_PEAK_HEIGHT))

# peak_indices, props = find_peaks(
#     ppg_filtered,
#     distance=int(0.55 * actual_fs),              # ~227 samples
#     prominence=0.3 * np.std(ppg_filtered),
#     width=(int(0.05 * actual_fs), int(0.25 * actual_fs)),
# )

peak_times = t[peak_indices]
rr = np.diff(peak_times)

print(f"\n── HRV Report (Filtered, before cleaning) ──")
print(f"Peaks detected:       {len(peak_indices)}")
print(f"RR intervals:         {len(rr)}")
print(f"Heart Rate:           {60.0 / np.mean(rr):.1f} bpm")
print(f"SDNN:                 {1000 * np.std(rr, ddof=1):.2f} ms")
print(f"RMSSD:                {1000 * np.sqrt(np.mean(np.diff(rr)**2)):.2f} ms")

# only consider signals between 0.45 and 1.2
# this is a BPM between 50-130
rr_clean = rr[(rr > 0.45) & (rr < 1.2)]

# calculate the median then only consider signals that are within 15% of the median
med_rr = np.median(rr_clean)
rr_clean = rr_clean[np.abs(rr_clean - med_rr) < 0.15 * med_rr]

mean_hr = 60.0 / np.mean(rr_clean)
rmssd = 1000 * np.sqrt(np.mean(np.diff(rr_clean)**2)) if len(rr_clean) > 2 else np.nan
max_hrv = 1000 * np.max(np.abs(np.diff(rr_clean))) if len(rr_clean) > 2 else np.nan

print(f"\n── HRV Report (Cleaned) ──")
print(f"RR intervals kept:    {len(rr_clean)} / {len(rr)}")
print(f"Heart Rate:           {mean_hr:.1f} bpm")
print(f"RMSSD:                {rmssd:.2f} ms")
print(f"Max HRV:              {max_hrv:.2f} ms")

ibi_ms = rr * 1000.0

# ── Figure 3: Filtered signal with detected peaks ──
fig3, ax = plt.subplots(figsize=(10, 4))
ax.plot(t, ppg_filtered, linewidth=0.8, color='tab:orange')
ax.plot(peak_times, ppg_filtered[peak_indices], 'ro', markersize=4, label='Peaks')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Voltage (V)")
ax.set_title(f"Filtered PPG + Peak Detection")
ax.legend()
ax.set_xlim(50,100)
ax.grid(True, alpha=0.3)
fig3.tight_layout()
plt.show()

# ── Figure 3b: Highlight regions with IBI > 1200 ms ──
IBI_THRESH_MS = 1200
fig3b, ax3b = plt.subplots(figsize=(10, 4))
ax3b.plot(t, ppg_filtered, linewidth=0.8, color='tab:orange')
ax3b.plot(peak_times, ppg_filtered[peak_indices], 'ro', markersize=4, label='Peaks')
for i, ibi in enumerate(ibi_ms):
    if ibi > IBI_THRESH_MS:
        ax3b.axvspan(peak_times[i], peak_times[i + 1], color='red', alpha=0.2,
                     label=f'IBI > {IBI_THRESH_MS} ms' if i == np.where(ibi_ms > IBI_THRESH_MS)[0][0] else None)
ax3b.set_xlabel("Time (s)")
ax3b.set_ylabel("Voltage (V)")
ax3b.set_title(f"Filtered PPG – Regions with IBI > {IBI_THRESH_MS} ms (likely missed peaks)")
ax3b.legend()
ax3b.grid(True, alpha=0.3)
ax3b.set_xlim(30,90)
fig3b.tight_layout()
plt.show()

# # ── Figure 4: Debug zoom around worst ΔIBI ──
# margin = 5
# zoom_lo = max(0, worst_time - margin)
# zoom_hi = min(t[-1], worst_time + margin)
# fig4, ax4 = plt.subplots(figsize=(10, 4))
# ax4.plot(t, ppg_filtered, linewidth=0.8, color='tab:orange')
# ax4.plot(peak_times, ppg_filtered[peak_indices], 'ro', markersize=6, label='Peaks')
# ax4.axvline(worst_time, color='red', linestyle='--', alpha=0.6, label=f'Max ΔIBI @ {worst_time:.2f}s')
# ax4.set_xlim(zoom_lo, zoom_hi)
# ax4.set_xlabel("Time (s)")
# ax4.set_ylabel("Voltage (V)")
# ax4.set_title(f"Debug: ±{margin}s around Max |ΔIBI| ({np.max(np.abs(successive_diff_ms)):.1f} ms)")
# ax4.legend()
# ax4.grid(True, alpha=0.3)
# fig4.tight_layout()
# plt.show()

# # ── Figure 5: Peak detection on raw signal (tuned parameters) ──
# wlen_samples = int(5.0 * actual_fs)
# min_prominence_raw = 0.05 * (ppg_raw.max() - ppg_raw.min())
# raw_peak_indices, _ = find_peaks(
#     ppg_raw,
#     distance=int(0.65 * actual_fs),                     # ~92 bpm max
#     prominence=0.4 * np.std(ppg_raw),
#     width=(int(0.08 * actual_fs), int(0.35 * actual_fs)),
#     wlen=int(1.5 * actual_fs),
# )
# raw_peak_times = t[raw_peak_indices]
# raw_ibi_s = np.diff(raw_peak_times)
# raw_ibi_ms = raw_ibi_s * 1000.0
# raw_successive_diff_ms = np.diff(raw_ibi_ms)
# raw_worst_idx = np.argmax(np.abs(raw_successive_diff_ms))
# raw_worst_time = raw_peak_times[raw_worst_idx + 1]

# print(f"\n── HRV Report (Raw Signal) ──")
# print(f"Peaks detected:       {len(raw_peak_indices)}")
# print(f"Mean IBI:             {np.mean(raw_ibi_ms):.1f} ms")
# print(f"Mean HR:              {60000.0 / np.mean(raw_ibi_ms):.1f} BPM")
# print(f"RMSSD:                {np.sqrt(np.mean(raw_successive_diff_ms**2)):.1f} ms")
# print(f"Max |ΔIBI|:           {np.max(np.abs(raw_successive_diff_ms)):.1f} ms  @ t={raw_worst_time:.2f}s")
# print(f"  IBI before:         {raw_ibi_ms[raw_worst_idx]:.1f} ms  ({60000/raw_ibi_ms[raw_worst_idx]:.1f} BPM)")
# print(f"  IBI after:          {raw_ibi_ms[raw_worst_idx+1]:.1f} ms  ({60000/raw_ibi_ms[raw_worst_idx+1]:.1f} BPM)")
# print(f"Mean |ΔIBI|:          {np.mean(np.abs(raw_successive_diff_ms)):.1f} ms")
# print(f"SDSD:                 {np.std(raw_successive_diff_ms):.1f} ms")

# fig5, ax5 = plt.subplots(figsize=(10, 4))
# ax5.plot(t, ppg_raw, linewidth=0.8)
# ax5.plot(raw_peak_times, ppg_raw[raw_peak_indices], 'ro', markersize=4, label=f'Peaks ({len(raw_peak_indices)})')
# ax5.set_xlabel("Time (s)")
# ax5.set_ylabel("Voltage (V)")
# ax5.set_title("Raw PPG + Peak Detection (same params)")
# ax5.legend()
# ax5.set_xlim(20, 90)
# ax5.grid(True, alpha=0.3)
# fig5.tight_layout()
# plt.show()

