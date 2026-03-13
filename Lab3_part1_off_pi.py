import os

import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "lab3_pi", "ppg_data.csv")
TARGET_FS_HZ = 500

# open csv which has form of time and raw ADC value
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
t = data[:, 0]
ppg_waveform = data[:, 1]
N = len(ppg_waveform)
dt = 1 / TARGET_FS_HZ
# remove DC offset (zero-mean)
ppg_demean = ppg_waveform - np.mean(ppg_waveform)
freq = np.fft.fftfreq(N, dt)
freq_pos = freq[1:N//2]

ppg_hamming = ppg_demean * np.hamming(N)
magnitude = np.abs(np.fft.fft(ppg_hamming)) / N
mag_pos = magnitude[1:N//2]
fundamental_index = 1 + np.argmax(mag_pos)
fundamental_freq_hz = freq[fundamental_index]
fundamental_mag = magnitude[fundamental_index]

# this is time domain and frequency domain without any filtering...
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 7))
# plot time domain for ax1
ax1.plot(t, ppg_waveform, linewidth=0.8)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax1.set_title(f"PPG – Time Domain")
ax1.grid(True, alpha=0.3)

# plot frequency domain for ax2
ax2.stem(freq_pos,mag_pos, label="Hamming")
ax2.scatter(fundamental_freq_hz, fundamental_mag, color='red', s=60, zorder=5, label=f'Fundamental ({fundamental_freq_hz:.4f} Hz)')
print("Fundamental frequency at", fundamental_freq_hz, "Hz -->", fundamental_freq_hz*60, "bpm")

ax2.set_xlabel('Frequency [Hz]')
ax2.set_ylabel('Magnitude')
ax2.set_title('Frequency Domain With Hamming Window')
ax2.legend()
ax2.set_xlim(0,10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()