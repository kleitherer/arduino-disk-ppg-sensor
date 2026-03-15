import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, ShortTimeFFT
from scipy.signal.windows import hann

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = "./ppg_data_rocking_finger_fast.csv"
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
ax2.set_title("PPG – Bandpass Filtered")
ax2.grid(True, alpha=0.3)

# subplot 3 – FFT of the filtered signal
ax3.plot(freq, mag, linewidth=0.8)
ax3.scatter(freq[top8_idx], mag[top8_idx], color='red', s=80, zorder=5,
            label='Top 8 peaks (0–2.5 Hz)')


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
ax3.set_title("FFT")
ax3.set_xlim(0, 4)
ax3.set_ylim(0, mag.max() * 1.5)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout(h_pad=3.0)
fig.savefig(os.path.join(SCRIPT_DIR, "ppg_time_and_freq.png"), dpi=150)
print("Saved → ppg_time_and_freq.png")

# STFT spectrogram
# Window length: 10 s → frequency resolution = 1/10 = 0.1 Hz, fine enough to
# separate beats ~0.05–0.1 Hz apart while still showing time evolution.
# Hop: 1 s → one new spectrum per second, giving smooth time axis.
win_1len  = int(10 * actual_fs)   # 10-second Hanning window
hop      = int(1  * actual_fs)   # 1-second hop (step between windows)
window   = hann(win_1len)

SFT = ShortTimeFFT(window, hop=hop, fs=actual_fs, mfft=win_1len)
Zxx = np.abs(SFT.stft(ppg_filtered))

stft_freqs = SFT.f                          # frequency axis
stft_times = SFT.t(len(ppg_filtered))       # time axis (centre of each window)

# Only show the 0–1 Hz band
freq_mask = (stft_freqs >= 0.2) & (stft_freqs <= 1.5)

fig2, ax = plt.subplots(figsize=(11, 4))
pcm = ax.pcolormesh(stft_times, stft_freqs[freq_mask],
                    Zxx[freq_mask, :], shading='gouraud', cmap='inferno')
fig2.colorbar(pcm, ax=ax, label='Magnitude')
ax.set_xlabel("Time (s)")
ax.set_ylabel("Frequency (Hz)")
ax.set_title("STFT Spectrogram – Bandpass Filtered PPG")
ax.set_ylim(0.4, 1.2)
ax.grid(True, alpha=0.2, color='white')

plt.tight_layout()
fig2.savefig(os.path.join(SCRIPT_DIR, "ppg_stft.png"), dpi=150)
print("Saved → ppg_stft.png")

plt.show()