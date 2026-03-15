import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(
    SCRIPT_DIR,
    "High-Precision-AD-DA-Board-Demo-Code",
    "RaspberryPI", "ADS1256", "python3",
))
import ADS1256
adc = ADS1256.ADS1256()

# Configuration
# PPG sensor "S" (signal) pin is wired into AIN7
PPG_CHANNEL = 6

# Front-end voltage gain (PGA).
# Start at gain 1 (widest range); increase if the signal is too small.
# Aim for a peak-to-peak swing of roughly ¼–½ of the ADC full scale.
#
#   Code │ Gain │ Full-scale range (Vref = 5 V)
#   ─────┼──────┼───────────────────────────────
#     0  │   1  │  ±5.000 V
#     1  │   2  │  ±2.500 V
#     2  │   4  │  ±1.250 V
#     3  │   8  │  ±0.625 V
#     4  │  16  │  ±0.313 V
#     5  │  32  │  ±0.156 V
#     6  │  64  │  ±0.078 V
GAIN_CODE = ADS1256.ADS1256_GAIN_E['ADS1256_GAIN_1']

# Hardware data rate.
# The ADS1256 has no native 250 SPS mode; the two nearest are 100 and 500 SPS.
# We set the chip to 500 SPS (one conversion every 2 ms) so there is always a
# fresh result ready before our 4 ms software deadline (= 250 Hz effective rate).
ADC_HW_DRATE = ADS1256.ADS1256_DRATE_E['ADS1256_1000SPS']

# Desired output sample rate
TARGET_FS_HZ = 500                    # current standards seem to be ~250 Hz for PPG! Let's see if it's good

# Total recording duration — collect this many seconds of raw PPG data.
# Increase for Part II (HRV needs several minutes of continuous data).
RECORD_DURATION_S = 60.0

# FFT window length — how many seconds of the recording to feed into the FFT.
# This sets frequency resolution:  Δf = 1 / FFT_WINDOW_S
#   10 s  →  Δf = 0.100 Hz  (~6 BPM resolution)
#   30 s  →  Δf = 0.033 Hz  (~2 BPM resolution)
#   62 s  →  Δf = 0.016 Hz  (~1 BPM resolution, resolves 60 vs 61 BPM)
# Must be ≤ RECORD_DURATION_S.  The FFT is taken from the END of the recording
# so any initial transient (finger settling) is excluded.
FFT_WINDOW_S = 10.0

# Output files
PLOT_TIME_PATH = os.path.join(SCRIPT_DIR, "ppg_time_domain_2.png")
PLOT_FFT_PATH  = os.path.join(SCRIPT_DIR, "ppg_fft.png")
CSV_PATH       = os.path.join(SCRIPT_DIR, "ppg_data.csv")

# Voltage Conversion
VREF           = 5.0       # volts (board reference)
ADC_FULL_SCALE = 0x7FFFFF  # 23-bit positive full scale for a 24-bit signed result


def raw_to_volts(raw: int, gain_code: int) -> float:
    """Convert a raw 24-bit ADS1256 reading to volts."""
    gain_factor = 1 << gain_code          # 1, 2, 4, … 64
    return raw * (VREF / gain_factor) / ADC_FULL_SCALE


# ADC Initialization

def setup_adc():
    if adc.ADS1256_init() != 0:
        raise RuntimeError("ADS1256 init failed – check wiring and SPI.")

    # Apply gain and hardware data rate in a single burst write
    adc.ADS1256_ConfigADC(GAIN_CODE, ADC_HW_DRATE)

    # ADS1256_ConfigADC hardcodes BUFEN=0. Re-enable the analog input buffer
    # (STATUS reg bit 1) so the ADC presents ~80 MΩ to the sensor instead of
    # loading it — same approach as Lab2's setup_adc_and_gpio.
    adc.ADS1256_WriteReg(0x00, 0x02)  # BUFEN=1

    # Fix MUX to PPG_CHANNEL once, then kick off continuous conversions.
    # The recording loop can then call ADS1256_Read_ADC_Data() directly,
    # skipping the per-read MUX change + SYNC + WAKEUP (~4-5 ms overhead).
    adc.ADS1256_SetChannal(PPG_CHANNEL)
    adc.ADS1256_WriteCmd(ADS1256.CMD['CMD_SYNC'])
    adc.ADS1256_WriteCmd(ADS1256.CMD['CMD_WAKEUP'])

    gain_factor = 1 << GAIN_CODE
    print(
        f"│ SW sample rate = {TARGET_FS_HZ} Hz"
    )


# Complete Recording

def recording(duration_s: float = RECORD_DURATION_S, fs_hz: int = TARGET_FS_HZ):
    """
    Collect PPG samples at `fs_hz` Hz for `duration_s` seconds.
    Returns (timestamps [s], voltages [V]).
    """
    print("Starting recording...")

    time.sleep(0.5) # wait for things to settle at the beginning
    n = int(duration_s * fs_hz) # total number of samples = total duration * number of samples/sec
                                # recording will stop when we reach this many samples     
    period = 1/fs_hz 
    time_array = np.empty(n)
    signal_v = np.empty(n) # signal in volts

    t0 = time.time() # returns current time (recording start time)

    for i in range(n):
        t_sample = t0 + i * period # sampling time for the i-th sample
        raw_value = adc.ADS1256_Read_ADC_Data()
        signal_v[i] = raw_to_volts(raw_value, GAIN_CODE)
        time_array[i] = time.time() - t0 # time since recording started

        sleep_time = t_sample + period - time.time()
        # If the ADC takes longer than the sampling period to read & convert the value,
        # then sleep_time will be negative. So, if sleep_time is negative, don't sleep!
        if sleep_time > 0:
            time.sleep(sleep_time)

    time_elapsed = time.time() - t0
    actual_rate = n / time_elapsed
    print(
        f"Done: Collected {n} samples in {time_elapsed:.2f} s,  (actual {actual_rate:.1f} Hz)\n"
        f"Signal range: {signal_v.min():.4f} V – {signal_v.max():.4f} V  "
    )

    return time_array, signal_v


def plot_time_domain(
    time_array: np.ndarray,
    signal_v: np.ndarray,
    path: str = PLOT_TIME_PATH,
) -> None:
    gain_factor = 1 << GAIN_CODE
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time_array, signal_v, linewidth=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.set_title(f"PPG – Time Domain  (PGA {gain_factor}×, fs = {TARGET_FS_HZ} Hz)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Time-domain plot → {path}")



def save_csv(time_array: np.ndarray, signal_v: np.ndarray, path: str = CSV_PATH) -> None:
    data = np.column_stack((time_array, signal_v))
    np.savetxt(path, data, delimiter=",", header="time_s,voltage_v", comments="")
    print(f"CSV saved → {path}")


def main():
    # setup_adc()
    # time_array, signal_v = recording()
    # plot_time_domain(time_array, signal_v)
    # save_csv(time_array, signal_v)


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
    gain_factor = 1 << GAIN_CODE
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(8, 7))
    # plot time domain for ax1
    ax1.plot(t, ppg_waveform, linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.set_title(f"PPG – Time Domain  (PGA {gain_factor}×, fs = {TARGET_FS_HZ} Hz)")
    ax1.grid(True, alpha=0.3)

    # plot frequency domain for ax2
    ax2.stem(freq_pos,mag_pos, label="Hamming")
    ax2.scatter(fundamental_freq_hz, fundamental_mag, color='red', s=60, zorder=5, label=f'Fundamental ({fundamental_freq_hz:.4f} Hz)')
    print("Fundamental frequency at", fundamental_freq_hz, "Hz -->", fundamental_freq_hz*60, "bpm")

    ax2.set_xlabel('Frequency [Hz]')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Frequency Domain With Hamming Window')
    # ax2.set_xlim(0, 3)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    main()
