# Arduino Disk PPG Sensing

**Course:** EE 217  
**Date:** [Date]

## Authors

1. Sayra Gorgani
2. Nadin Souki
3. Kaitlyn Leitherer

---

## Section 1 - Taking HR Readings in Time and Frequency Domains with PPG sensor


We chose an ADC output rate (F_s) of 250Hz based on literature using PPG and our assumption that 
the frequency we'll be measuring will lie between 1-2Hz (assuming that 60-100 beats per minute is
the range of human resting heart rate).



Taking your sample stream, run it through an FFT of appropriate length (what length did you
choose?), and do not forget to apply windowing before the FFT. Remember that you are
trying to measure heart rate, so the difference (in frequency) of 60 bpm and 61 bpm is a
whopping .016 Hz. You will need a fairly long stream of samples to accurately measure this;
by “appropriate length”, the length of the FFT (number of samples in time) will determine
your frequency resolution. So PPG is about measuring extremely tiny changes in frequency,
which is why the FFT is your best tool for doing this. Plot the FFT output for your finger. 
Are there any peaks in the FFT that are not harmonics of the fundamental?

number of samples in time... need about 10s



## Section 2

[Content]

## Section 3

[Content]
