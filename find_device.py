import subprocess
import signal
import time
import numpy as np
from scipy.io import wavfile
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

RATE = 48000
SECONDS = 3
OUTPUT_FILENAME = "output.wav"
TARGET = "alsa_input.usb-Yamaha_Corporation_Steinberg_UR816C-00.pro-input-0"

print(f"Aufnahme: 10 Kanäle (extrahiere 0-3), {RATE} Hz, {SECONDS}s ...")

proc = subprocess.Popen([
    "pw-record",
    "--channels", "10",       # Alle 10 Kanäle aufnehmen
    "--rate",     str(RATE),
    "--format",   "s32",
    "--target",   TARGET,
    OUTPUT_FILENAME
])

time.sleep(SECONDS)
proc.send_signal(signal.SIGINT)
proc.wait()
print("Fertig.")

# === WAV laden ===
fs_rate, audio_signal = wavfile.read(OUTPUT_FILENAME)
print(f"Shape: {audio_signal.shape} | Rate: {fs_rate} Hz | dtype: {audio_signal.dtype}")

# Kanäle 0-3 = physische Inputs 1-4
mikro_1 = audio_signal[:, 0]
mikro_2 = audio_signal[:, 1]
mikro_3 = audio_signal[:, 2]
mikro_4 = audio_signal[:, 3]

for i, m in enumerate([mikro_1, mikro_2, mikro_3, mikro_4]):
    print(f"Mic {i+1} peak: {np.max(np.abs(m))}")

# === Plot ===
N = len(audio_signal)
t = np.linspace(0, N / fs_rate, num=N)
colors = ["blue", "orange", "green", "red"]
mics   = [mikro_1, mikro_2, mikro_3, mikro_4]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i in range(4):
    axes[i].plot(t, mics[i], color=colors[i])
    axes[i].set_title(f"Mikrofon {i+1} (Kanal {i+1})")
    axes[i].set_xlabel("Zeit [s]")
    axes[i].set_ylabel("Amplitude")
    axes[i].grid()

plt.tight_layout()
plt.show()

#plot all signals in one plot
plt.figure(figsize=(12, 6))
for i in range(4):
    plt.plot(t, mics[i], label=f"Mikrofon {i+1}", color=colors[i])
plt.title("Alle Mikrofone")
plt.xlabel("Zeit [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()