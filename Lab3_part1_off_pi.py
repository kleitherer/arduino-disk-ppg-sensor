import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "./ppg_data_500hz_50s.csv"
TARGET_FS_HZ = 500

# ── Load data ────────────────────────────────────────────────────────────────
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
t = data[:, 0]
ppg_raw = data[:, 1]
N = len(ppg_raw)
actual_fs = (N - 1) / (t[-1] - t[0])
dt = 1 / actual_fs
print(f"N = {N}, actual fs = {actual_fs:.2f} Hz")

# ── Bandpass filter: 0.5 – 4 Hz ──────────────────────────────────────────────
# Removes DC drift (below 0.5 Hz) and high-frequency noise (above 4 Hz).
# Heart rate spans 0.5–2 Hz fundamental plus several harmonics up to ~4 Hz.
b, a = butter(4, [0.5, 4.0], btype='bandpass', fs=actual_fs)
ppg_filtered = filtfilt(b, a, ppg_raw)

# ── FFT of filtered + Hanning-windowed signal ─────────────────────────────────
ppg_windowed = ppg_filtered * np.hanning(N)
freq = np.fft.rfftfreq(N, dt)
mag  = np.abs(np.fft.rfft(ppg_windowed)) / N

# Locate the top 8 peaks in the 0–2.5 Hz band
hr_band = (freq >= 0.0) & (freq <= 2.5)
masked_mag = np.where(hr_band, mag, 0.0)
top8_idx  = np.argsort(masked_mag)[-8:][::-1]
fund_idx  = top8_idx[0]
fund_freq = freq[fund_idx]
fund_mag  = mag[fund_idx]

print(f"Fundamental: {fund_freq:.4f} Hz  →  {fund_freq * 60:.1f} BPM")
print("Top 8 peaks (0–2.5 Hz):")
for i, idx in enumerate(top8_idx):
    print(f"  {i+1}. {freq[idx]:.4f} Hz ({freq[idx]*60:.1f} BPM), mag={mag[idx]:.4f}")

# ── Figure 1: time domain (raw + filtered) and FFT ───────────────────────────
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(10, 12))

# subplot 1 – raw time domain
ax1.plot(t, ppg_raw, linewidth=0.6)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("PPG – Raw Time Domain")
ax1.grid(True, alpha=0.3)

# subplot 2 – bandpass-filtered time domain
ax2.plot(t, ppg_filtered, linewidth=0.8, color='darkorange')
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Amplitude (V)")
ax2.set_title("PPG – Bandpass Filtered (0.5–4 Hz)")
ax2.grid(True, alpha=0.3)

# subplot 3 – FFT of the filtered signal
ax3.plot(freq, mag, linewidth=0.8)
ax3.scatter(freq[top8_idx], mag[top8_idx], color='red', s=80, zorder=5,
            label='Top 8 peaks (0–2.5 Hz)')

# Annotate each peak, staggering label height to avoid overlap
sorted_by_freq = top8_idx[np.argsort(freq[top8_idx])]
y_max = mag.max()
prev_x, prev_y_text = -np.inf, 0.0
min_x_gap = 0.08
offsets = [14, 28, 42]
offset_cycle = 0
for idx in sorted_by_freq:
    x, y = freq[idx], mag[idx]
    if (x - prev_x) < min_x_gap:
        offset_cycle = (offset_cycle + 1) % len(offsets)
    else:
        offset_cycle = 0
    y_offset = offsets[offset_cycle]
    ax3.annotate(f'{x:.3f} Hz', xy=(x, y),
                 xytext=(0, y_offset), textcoords='offset points',
                 ha='center', fontsize=7,
                 arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
    prev_x = x

ax3.set_xlabel("Frequency (Hz)")
ax3.set_ylabel("Magnitude")
ax3.set_title("FFT – Bandpass Filtered + Hanning Window")
ax3.set_xlim(0, 4)
ax3.set_ylim(0, mag.max() * 1.5)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "ppg_time_and_freq.png"), dpi=150)
print("Saved → ppg_time_and_freq.png")
plt.show()