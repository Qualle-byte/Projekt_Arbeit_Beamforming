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

from taucompensation import FRAME_RATE

matplotlib.use("TkAgg")
np.set_printoptions(threshold=sys.maxsize)

RATE = 48000
SECONDS = 2
OUTPUT_FILENAME = "output.wav"
TARGET = "alsa_input.usb-Yamaha_Corporation_Steinberg_UR816C-00.pro-input-0"

# Pilot signal
np.random.seed(42)  # For reproducibility
bit_Signal = np.random.randint(0,2, size=1024)
Pilot_signal = np.zeros(1024)
for i in range(len(bit_Signal)):
    if bit_Signal[i] == 0:
        Pilot_signal[i] = int(1)
    else:
        Pilot_signal[i] = int(-1)
    np.append(Pilot_signal, Pilot_signal[i])

pilot_16bit = np.int16(Pilot_signal * 32767)
# 2 kHz sine wave - DIESMAL 1 SEKUNDE LANG (48000 Samples)
sine_wave = np.sin(2 * np.pi * 2000 * np.arange(48000) / 48000)

# append sine wave to Pilot_signal
Signal_A = np.append(Pilot_signal, sine_wave)

# Auf 16-Bit skalieren (wie zuvor besprochen)
audio_data_16bit = np.int16(Signal_A * 32767)

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

# === 1. Signale normieren und Kreuzkorrelation berechnen ===

# Wir definieren eine kleine Funktion, die das Normieren und Korrelieren übernimmt
def finde_startpunkt(mic_signal, pilot):
    # 1. Mikrofonsignal normieren (Wertebereich -1.0 bis 1.0)
    norm_mic = mic_signal / np.max(np.abs(mic_signal))
    
    # 2. Kreuzkorrelation mit dem reinen Pilot_signal (-1.0 und 1.0)
    korrelation = np.correlate(norm_mic, pilot, mode='valid')
    
    # 3. Index mit der höchsten Übereinstimmung zurückgeben
    return np.argmax(np.abs(korrelation))

# Startpunkte für alle 4 Mikrofone berechnen
start_idx_1 = finde_startpunkt(mikro_1, Pilot_signal)
start_idx_2 = finde_startpunkt(mikro_2, Pilot_signal)
start_idx_3 = finde_startpunkt(mikro_3, Pilot_signal)
start_idx_4 = finde_startpunkt(mikro_4, Pilot_signal)

print(f"Startpunkt bei Mic 1 gefunden bei Sample: {start_idx_1} (Zeit: {start_idx_1 / RATE:.4f} s)")
print(f"Startpunkt bei Mic 2 gefunden bei Sample: {start_idx_2} (Zeit: {start_idx_2 / RATE:.4f} s)")
print(f"Startpunkt bei Mic 3 gefunden bei Sample: {start_idx_3} (Zeit: {start_idx_3 / RATE:.4f} s)")
print(f"Startpunkt bei Mic 4 gefunden bei Sample: {start_idx_4} (Zeit: {start_idx_4 / RATE:.4f} s)")
# === 2. Signale durch Slicing trennen ===
laenge_pilot = 1024
laenge_sinus = 48000  # (Oder 1024, je nachdem wie lang dein Sinus aktuell ist)

# Herausschneiden des aufgenommenen BPSK-Signals
rec_pilot_1 = mikro_1[start_idx_1 : start_idx_1 + laenge_pilot]

# Herausschneiden des aufgenommenen Sinus (startet direkt nach dem Pilot)
rec_sinus_1 = mikro_1[start_idx_1 + laenge_pilot : start_idx_1 + laenge_pilot + laenge_sinus]

# === 3. Zur Kontrolle Plotten ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Zeitachse für BPSK (Startet bei 0 Sekunden)
t_pilot = np.arange(len(rec_pilot_1)) / RATE
axes[0].plot(t_pilot, rec_pilot_1, color='blue')
axes[0].set_title("Extrahiertes BPSK-Pilotsignal")
axes[0].set_xlabel("Zeit ab Signalstart [s]")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

# Zeitachse für Sinus dynamisch an die echte Array-Länge anpassen
t_sinus = (np.arange(len(rec_sinus_1)) + laenge_pilot) / RATE
axes[1].plot(t_sinus, rec_sinus_1, color='green')
axes[1].set_title("Extrahierte Sinuswelle (Komplett)")
axes[1].set_xlabel("Zeit ab Signalstart [s]")
axes[1].grid()

plt.tight_layout()
plt.show()
# Herausschneiden des aufgenommenen BPSK-Signals
rec_pilot_2 = mikro_1[start_idx_2 : start_idx_2 + laenge_pilot]

# Herausschneiden des aufgenommenen Sinus (startet direkt nach dem Pilot)
rec_sinus_2 = mikro_1[start_idx_2 + laenge_pilot : start_idx_2 + laenge_pilot + laenge_sinus]

# === 3. Zur Kontrolle Plotten ===
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Zeitachse für BPSK (Startet bei 0 Sekunden)
t_pilot = np.arange(len(rec_pilot_2)) / RATE
axes[0].plot(t_pilot, rec_pilot_2, color='blue')
axes[0].set_title("Extrahiertes BPSK-Pilotsignal")
axes[0].set_xlabel("Zeit ab Signalstart [s]")
axes[0].set_ylabel("Amplitude")
axes[0].grid()

# Zeitachse für Sinus dynamisch an die echte Array-Länge anpassen
t_sinus = (np.arange(len(rec_sinus_2)) + laenge_pilot) / RATE
axes[1].plot(t_sinus, rec_sinus_2, color='green')
axes[1].set_title("Extrahierte Sinuswelle (Komplett)")
axes[1].set_xlabel("Zeit ab Signalstart [s]")
axes[1].grid()

plt.tight_layout()
plt.show()