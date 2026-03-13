import os

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "lab3_pi", "ppg_data_500hz_60s.csv")
TARGET_FS_HZ = 500

# open csv which has columns: time_s, voltage_v
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
t = data[:, 0]
ppg_waveform = data[:, 1]
N = len(ppg_waveform)
actual_fs = (N - 1) / (t[-1] - t[0])
dt = 1 / actual_fs
print(f"N = {N}, actual fs = {actual_fs:.2f} Hz")
freq = np.fft.fftfreq(N, dt)
freq_pos = freq[1:N//2]

# raw FFT (no DC removal, no windowing)
mag_raw = np.abs(np.fft.fft(ppg_waveform)) / N
mag_raw_pos = mag_raw[1:N//2]
raw_fund_idx = 1 + np.argmax(mag_raw_pos)
raw_fund_freq = freq[raw_fund_idx]
raw_fund_mag = mag_raw[raw_fund_idx]

# DC-removed + Hamming window FFT
ppg_demean = ppg_waveform - np.mean(ppg_waveform)
ppg_hanning = ppg_demean * np.hanning(N)
mag_ham = np.abs(np.fft.fft(ppg_hanning)) / N
mag_ham_pos = mag_ham[1:N//2]
ham_fund_idx = 1 + np.argmax(mag_ham_pos)
ham_fund_freq = freq[ham_fund_idx]
ham_fund_mag = mag_ham[ham_fund_idx]

print("Raw fundamental:", raw_fund_freq, "Hz -->", raw_fund_freq * 60, "bpm")
print("Hamming fundamental:", ham_fund_freq, "Hz -->", ham_fund_freq * 60, "bpm")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=False, figsize=(8, 10))

# subplot 1: time domain
ax1.plot(t, ppg_waveform, linewidth=0.8)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title("PPG – Time Domain")
ax1.grid(True, alpha=0.3)

# subplot 2: frequency domain — raw (no DC removal, no windowing)
ax2.stem(freq_pos, mag_raw_pos, label="No windowing")
ax2.scatter(raw_fund_freq, raw_fund_mag, color='red', s=60, zorder=5,
            label=f'Fundamental ({raw_fund_freq:.4f} Hz)')
ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Domain – Raw (no DC removal, no window)')
ax2.legend()
ax2.set_xlim(0, 10)
ax2.grid(True, alpha=0.3)

# subplot 3: frequency domain — DC-removed + Hamming window
ax3.stem(freq_pos, mag_ham_pos, label="Hanning")
ax3.scatter(ham_fund_freq, ham_fund_mag, color='red', s=60, zorder=5,
            label=f'Fundamental ({ham_fund_freq:.4f} Hz)')
ax3.set_xlabel('Frequency [Hz]')
ax3.set_ylabel('Magnitude')
ax3.set_title('Frequency Domain – DC Removed + Haming Window')
ax3.legend()
ax3.set_xlim(0, 15)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(SCRIPT_DIR, "ppg_time_and_freq.png"), dpi=150)
print(f"Saved → ppg_time_and_freq.png")
plt.show()

# --- Figure 2: Reconstruct signal after zeroing out 1.5–1.7 Hz and 2.7–2.9 Hz ---
fft_full = np.fft.fft(ppg_waveform)
fft_filtered = fft_full.copy()
abs_freq = np.abs(freq)
zero_mask = (((abs_freq >= 1.5) & (abs_freq <= 1.7)) |
             ((abs_freq >= 2.7) & (abs_freq <= 2.9)))
fft_filtered[zero_mask] = 0
ppg_reconstructed = np.fft.ifft(fft_filtered).real

fig2, (ax_orig, ax_recon) = plt.subplots(2, 1, sharex=True, figsize=(8, 7))

ax_orig.plot(t, ppg_waveform, linewidth=0.8)
ax_orig.set_ylabel("Voltage (V)")
ax_orig.set_title("Original PPG – Time Domain")
ax_orig.grid(True, alpha=0.3)

ax_recon.plot(t, ppg_reconstructed, linewidth=0.8, color='tab:orange')
ax_recon.set_xlabel("Time (s)")
ax_recon.set_ylabel("Voltage (V)")
ax_recon.set_title("Reconstructed PPG – 1.5–1.7 Hz & 2.7–2.9 Hz Zeroed Out")
ax_recon.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(SCRIPT_DIR, "ppg_reconstructed.png"), dpi=150)
print(f"Saved → ppg_reconstructed.png")
plt.show()