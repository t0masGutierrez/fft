import numpy as np
from numpy import fft
import matplotlib.pyplot as plt

# sample frequency (Hz)
f = 1000

# sample time (s)
dt = 1.0

# number of samples
num = int(f*dt)

# total time (s)
t = np.linspace(0, dt, num, endpoint=False)

# generate signal
f1, f2 = 50, 120
signal = 2 * np.cos(2 * np.pi * f1 * t) + 3 * np.cos(2 * np.pi * f2 * t)

# perform fft
fftSignal = np.fft.fft(signal)
fftFreq = np.fft.fftfreq(num, d=1/f)

# get magnitude
N = len(signal)
magnitude = (2/N) * np.abs(fftSignal)[:int(N/2)]
freqs = fftFreq[:int(N/2)]

# get phase
phase = np.angle(fftSignal[:int(N/2)])
threshold = 1e-10
phase[magnitude < threshold] = 0

# plot
fig, axs = plt.subplots(2, 1, figsize=(10, 6))

# plot magnitude
axs[0].plot(freqs, magnitude)
axs[0].set_xlabel("frequency (Hz)")
axs[0].set_ylabel("Amplitude")
axs[0].set_title("Fast Fourier Transform")
axs[0].grid(True)
axs[0].set_xlim(0, f1 + f2)

# plot phase
axs[1].plot(freqs, phase)
axs[1].set_xlabel("frequency (Hz)")
axs[1].set_ylabel("Phase (radians)")
axs[1].set_title("Phase Spectrum")
axs[1].grid(True)
axs[1].set_xlim(0, f1 + f2)

plt.tight_layout()
plt.show()
