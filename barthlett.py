# --- Bibliotheken einbinden ---
import Funktionen
import wave  # Zum Speichern der Aufnahme im WAV-Format
import matplotlib
import math
import matplotlib.pyplot as plt  # Zum Erzeugen von Diagrammen
import numpy as np  # Für numerische Operationen (z. B. Arrays, Mittelung, linspace)
import pyaudio  # Für Audioaufnahme über das Mikrofon
from scipy import signal  # Für die Berechnung des Spektrogramms
from scipy.fft import (
    fft,
    fftfreq,
)  # Für die Fouriertransformation (Spektralanalyse)
from scipy.io import wavfile  # Zum Einlesen von WAV-Dateien

matplotlib.use(
    "TkAgg"
)  # GUI-Backend für Matplotlib (wird für interaktive Fensterdarstellung benötigt)

# === Aufnahme-Parameter definieren ===
# === Correct parameters for UR816C ===
FRAMES_PER_BUFFER = 4800        # Chunk size (0.1s at 48kHz)
FORMAT = pyaudio.paInt32        # 32-bit — must match s32le from PipeWire
CHANNELS = 4                    # Physical inputs 1–4 on the UR816C
RATE = 48000                    # Must be 48000 — what the device actually runs at
RECORD_SECONDS = 10
OUTPUT_FILENAME = "output.wav"
DEVICE_INDEX = 2                # UR816C index found by scanner

# === Aufnahme starten ===
# === WAV einlesen & Kanäle trennen ===
fs_rate, audio_signal = wavfile.read("10grad.wav")
mikro_1 = audio_signal[:, 0]   # Physical input 1
mikro_2 = audio_signal[:, 1]   # Physical input 2
mikro_3 = audio_signal[:, 2]   # Physical input 3
mikro_4 = audio_signal[:, 3]   # Physical input 4


# === Vorbereitung für Zeitbereichsanalyse ===
N = len(audio_signal)  # Anzahl der Samples
t = np.linspace(0, N / fs_rate, num=N)  # Zeitachse in Sekunden

## === Zeitbereichs-Darstellung des Signals ===
# Funktionen.Zeitbereich_Plot(t,mikro_1,mikro_2,mikro_3,mikro_4)
Funktionen.Zeitbereich_alle_in_einem_Plot(t,mikro_1,mikro_2,mikro_3,mikro_4)

# === Frequenzbereichsanalyse mit FFT ===
FFT_mikro_1 = np.abs(fft(mikro_1))  # Betrag der komplexen Fouriertransformation
freqs = fftfreq(N, 1 / fs_rate)  # Frequenzachse berechnen
FFT_mikro_2 = np.abs(fft(mikro_2))  # Betrag der komplexen Fouriertransformation
freqs = fftfreq(N, 1 / fs_rate)  # Frequenzachse berechnen



# 1. Finde den Index der maximalen Frequenz im Spektrum (sollte bei ca. 2000 Hz liegen)
# Wir suchen nur in der ersten Hälfte (positive Frequenzen)
idx_peak = np.argmax(FFT_mikro_1[:N//2])
freq_peak = freqs[idx_peak]
#print(f"Dominante Frequenz gefunden bei: {freq_peak:.2f} Hz")

# 2. Kreuzspektrum NUR für diese Frequenz auswerten
x_peak = fft(mikro_1)[idx_peak]
y_peak = fft(mikro_2)[idx_peak]


# 3. Winkel über Leistungsdichtespektrum berechnen
c = 343.0  # Schallgeschwindigkeit in m/s
d = 0.08575    # Mikrofonabstand in m
Funktionen.Winkel_LDS(x_peak, y_peak,freq_peak,d)
mikro_daten = audio_signal[:, 0:4]
normierte_mikro_daten = Funktionen.normiertefunktionen(mikro_daten)
mikro_daten_analytic = signal.hilbert(normierte_mikro_daten, axis=0)

# Autokovarianzmatrix
R_xx = Funktionen.Autokovarianzmatrix_schneller(mikro_daten_analytic,N)
R_xx1 = np.cov(mikro_daten_analytic, rowvar=False)  # Autokovarianzmatrix für die ersten 4 Mikrofone
#R_xx1 = (mikro_daten.conj().T @ mikro_daten) / N



#Bartlett
lamda = c/freq_peak
beta= (2 * np.pi)/lamda
m_indices=np.linspace(0,3,4)
phi_deg = np.arange(0,180,1)
phi_rad = np.radians(phi_deg)

P_max=-np.inf

P_values = []
for x in np.nditer(phi_rad):
    a= np.exp(-1j * beta* d * m_indices * np.cos(x))
    P_Barthlet = np.real(a.conj().T @ R_xx1 @ a).item()
    P_values.append(P_Barthlet)

    if P_Barthlet > P_max:
        P_max = P_Barthlet
        final_angle = float(x)

print("Das ist der finale echte Winkel mittels Bartlett:")
print(np.degrees (final_angle))
P_values = np.array(P_values)
# In dB umrechnen für eine schönere Darstellung
P_values_dB = 10 * np.log10(P_values / np.max(P_values))

best_angle_idx = np.argmax(P_values_dB)
final_angle = phi_deg[best_angle_idx]

plt.figure(figsize=(10, 5))
plt.plot(phi_deg, P_values_dB, label="Bartlett Spektrum", color='b', linewidth=2)
plt.axvline(final_angle, color='r', linestyle='--', label=f'Gefundener Winkel: {final_angle}°')
plt.title(f"DOA (Zeitbereichs-Kovarianz) gefiltert auf {freq_peak:.0f} Hz")
plt.xlabel("Winkel [Grad]")
plt.ylabel("Leistung [dB]")
plt.xlim(0, 180)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(phi_deg, P_values, label="Bartlett Spektrum", color='b', linewidth=2)
plt.axvline(final_angle, color='r', linestyle='--', label=f'Gefundener Winkel: {final_angle}°')
plt.title(f"DOA (Zeitbereichs-Kovarianz) gefiltert auf {freq_peak:.0f} Hz")
plt.xlabel("Winkel [Grad]")
plt.ylabel("Leistung") 
plt.xlim(0, 180)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

Funktionen.plot_polar_spectrum(phi_deg, P_values_dB, final_angle, freq_peak)