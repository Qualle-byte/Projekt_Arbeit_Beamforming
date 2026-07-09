import subprocess
import signal
import time
import wave
import numpy as np
from scipy.io import wavfile
import matplotlib
import matplotlib.pyplot as plt
import sys
import simpleaudio as sa

matplotlib.use("TkAgg")
np.set_printoptions(threshold=sys.maxsize)

RATE = 48000
SECONDS = 3
OUTPUT_FILENAME = "output.wav"
TARGET = "alsa_input.usb-Yamaha_Corporation_Steinberg_UR816C-00.pro-input-0"

# Pilot signal
bit_Signal = np.random.randint(0,2, size=1024)
Pilot_signal = np.zeros(1024)
for i in range(len(bit_Signal)):
    if bit_Signal[i] == 0:
        Pilot_signal[i] = int(1)
    else:
        Pilot_signal[i] = int(-1)
    np.append(Pilot_signal, Pilot_signal[i])

# 2 kHz sine wave
sine_wave = np.sin(2 * np.pi * 2000 * np.arange(1024) / 48000)
# append sine wave to Pilot_signal
Signal_A = np.append(Pilot_signal, sine_wave)
    
FRAME_RATE = 48000
with wave.open("Pilot_signal.wav", mode="wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(1)
    wav_file.setframerate(FRAME_RATE)
    wav_file.writeframes(bytes(Signal_A))
    wav_file.close()


print(f"Aufnahme: 10 Kanäle (extrahiere 0-3), {RATE} Hz, {SECONDS}s ...")

# 1. Aufnahme im Hintergrund starten
proc = subprocess.Popen([
    "pw-record",
    "--channels", "10",       
    "--rate",     str(RATE),
    "--format",   "s32",
    "--target",   TARGET,
    OUTPUT_FILENAME
])

# 2. Kurze Pause, damit die Aufnahme sicher läuft, bevor der Ton startet
time.sleep(0.2)

# 3. Ton abspielen (läuft im Hintergrund weiter)
print("Spiele Pilot-Signal ab...")
subprocess.run(["pw-play", "Pilot_signal.wav"])
print("Abspielen beendet.")

# 4. Restliche Zeit abwarten (SECONDS minus die Pause oben)
time.sleep(SECONDS - 0.2)

# 5. Aufnahme beenden
proc.send_signal(signal.SIGINT)
proc.wait()
print("Fertig.")


# wav ploten
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

# plot all signals in one plot
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