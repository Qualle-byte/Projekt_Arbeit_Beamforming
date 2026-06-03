# --- Bibliotheken einbinden ---
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
FRAMES_PER_BUFFER = 3200  # Größe eines Datenblocks (Chunk): 3200 Samples werden auf einmal gelesen
FORMAT = (
    pyaudio.paInt16
)  # Audioformat: 16 Bit pro Sample (Standard bei CD-Qualität)
CHANNELS = 2  # Stereo-Aufnahme (2 Kanäle)
RATE = 16000  # Abtastrate in Hz: 16.000 Samples pro Sekunde
RECORD_SECONDS = 10  # Dauer der Aufnahme in Sekunden
OUTPUT_FILENAME = "output.wav"  # Dateiname der gespeicherten Aufnahme

## === Aufnahme starten ===
#p = pyaudio.PyAudio()  # Initialisiere das PyAudio-Objekt
#stream = p.open(
    #format=FORMAT,
    #channels=CHANNELS,
    #rate=RATE,
    #input=True,
    #frames_per_buffer=FRAMES_PER_BUFFER,
#)

#print("Recording...")

#frames = []  # Liste zum Zwischenspeichern der aufgenommenen Datenblöcke

## Schleife liest kontinuierlich Chunks und speichert sie
#for _ in range(0, int(RATE / FRAMES_PER_BUFFER * RECORD_SECONDS)):
    #data = stream.read(FRAMES_PER_BUFFER)
    #frames.append(data)

## === Aufnahme beenden und Stream schließen ===
#stream.stop_stream()
#stream.close()
#p.terminate()

## === Aufnahme in WAV-Datei speichern ===
#with wave.open(OUTPUT_FILENAME, "wb") as wf:
    #wf.setnchannels(CHANNELS)  # Anzahl Kanäle setzen (Stereo)
    #wf.setsampwidth(p.get_sample_size(FORMAT))  # Sample-Breite in Bytes
    #wf.setframerate(RATE)  # Abtastrate setzen
    #wf.writeframes(
        #b"".join(frames)
    #)  # Alle aufgenommenen Blöcke zusammenfügen und schreiben

# === WAV-Datei erneut einlesen zur Analyse ===
fs_rate, audio_signal = wavfile.read(OUTPUT_FILENAME)

mikro_1 = audio_signal[:, 0]  # Linker Kanal
mikro_2 = audio_signal[:, 1]  # Rechter Kanal

# Falls die Aufnahme Stereo ist, beide Kanäle zu Mono mitteln
# if audio_signal.ndim == 2:
# audio_signal = audio_signal.mean(axis=1)

# === Vorbereitung für Zeitbereichsanalyse ===
N = len(audio_signal)  # Anzahl der Samples
t = np.linspace(0, N / fs_rate, num=N)  # Zeitachse in Sekunden

## === Zeitbereichs-Darstellung des Signals ===
# plt.figure(figsize=(15, 5))
# plt.plot(t, audio_signal)
# plt.title("Audio Signal in Time Domain")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.tight_layout()
# plt.grid()

# Plot für Mikrofon 1
plt.subplot(2, 1, 1)
plt.plot(t, mikro_1, color="blue")
plt.title("Mikrofon 1 (Linker Kanal)")
plt.xlabel("Zeit [s]")
plt.ylabel("Amplitude")
plt.grid()

# Plot für Mikrofon 2
plt.subplot(2, 1, 2)
plt.plot(t, mikro_2, color="orange")
plt.title("Mikrofon 2 (Rechter Kanal)")
plt.xlabel("Zeit [s]")
plt.ylabel("Amplitude")
plt.grid()

plt.tight_layout()
plt.show()

# === Frequenzbereichsanalyse mit FFT ===
FFT_mikro_1 = np.abs(fft(mikro_1))  # Betrag der komplexen Fouriertransformation
freqs = fftfreq(N, 1 / fs_rate)  # Frequenzachse berechnen
FFT_mikro_2 = np.abs(fft(mikro_2))  # Betrag der komplexen Fouriertransformation
freqs = fftfreq(N, 1 / fs_rate)  # Frequenzachse berechnen

plt.figure(figsize=(15, 8))

# --- Darstellung: vollständiges (symmetrisches) Spektrum ---
plt.subplot(2, 1, 1)
plt.plot(freqs, FFT_mikro_1)
plt.title("FFT - Double-sided Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()

# --- Darstellung: nur positive Frequenzen (nützlicher Teil) ---
plt.subplot(2, 1, 2)
plt.plot(freqs[: N // 2], FFT_mikro_1[: N // 2])
plt.title("FFT - Single-sided Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()

plt.subplot(2, 1, 1)
plt.plot(freqs, FFT_mikro_2)
plt.title("FFT - Double-sided Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()

# --- Darstellung: nur positive Frequenzen (nützlicher Teil) ---
plt.subplot(2, 1, 2)
plt.plot(freqs[: N // 2], FFT_mikro_2[: N // 2])
plt.title("FFT - Single-sided Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid()
plt.tight_layout()
plt.show()

# === Spektrogramme für BEIDE Mikrofone erzeugen ===
plt.figure(figsize=(15, 10))

# --- 1. Spektrogramm für Mikrofon 1 ---
plt.subplot(2, 1, 1)
# STFT berechnen für Mikro 1
f1, t_spec1, Sxx1 = signal.spectrogram(mikro_1, fs=fs_rate, mode="complex")
phase1 = np.angle(Sxx1)  # Phase extrahieren 
Sxx_db1 = 10 * np.log10(Sxx1 + 1e-12)
Sxx_db1 -= np.max(Sxx_db1)  # Auf 0 dBFS normieren

# --- 1. Berechnungen für Mikrofon 1 ---
f1, t_spec1, Sxx1_complex = signal.spectrogram(
    mikro_1, fs=fs_rate, mode="complex"
)
phase1 = np.angle(Sxx1_complex)  # Phase extrahieren (float)

# Betrag/Leistung korrekt berechnen vor dem Logarithmus!
Sxx1_magnitude = np.abs(Sxx1_complex) ** 2
Sxx_db1 = 10 * np.log10(Sxx1_magnitude + 1e-12)
Sxx_db1 -= np.max(Sxx_db1)  # Auf 0 dBFS normieren

# --- 2. Berechnungen für Mikrofon 2 ---
f2, t_spec2, Sxx2_complex = signal.spectrogram(
    mikro_2, fs=fs_rate, mode="complex"
)
phase2 = np.angle(Sxx2_complex)  # Phase extrahieren (float)

# Betrag/Leistung korrekt berechnen vor dem Logarithmus!
Sxx2_magnitude = np.abs(Sxx2_complex) ** 2
Sxx_db2 = 10 * np.log10(Sxx2_magnitude + 1e-12)
Sxx_db2 -= np.max(Sxx_db2)  # Auf 0 dBFS normieren


# --- 3. Visualisierung (2x2 Grid für perfekte Übersicht) ---
plt.figure(figsize=(15, 10))

# --- MIKROFON 1 ---
# Links: Leistung
plt.subplot(2, 2, 1)
pcm1_db = plt.pcolormesh(
    t_spec1, f1, Sxx_db1, shading="gouraud", cmap="viridis"
)
plt.ylabel("Frequenz [Hz]")
plt.title("Spektrogramm (Leistung) - Mikro 1")
plt.colorbar(pcm1_db, label="Spektrale Leistung [dB]")

# Rechts: Phase
plt.subplot(2, 2, 2)
pcm1_ph = plt.pcolormesh(
    t_spec1, f1, phase1, shading="gouraud", cmap="twilight"
)
plt.ylabel("Frequenz [Hz]")
plt.title("Phasen-Spektrogramm - Mikro 1")
cbar1 = plt.colorbar(
    pcm1_ph,
    label="Phase [Radiant]",
    ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
)
cbar1.ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

# --- MIKROFON 2 ---
# Links: Leistung
plt.subplot(2, 2, 3)
pcm2_db = plt.pcolormesh(
    t_spec2, f2, Sxx_db2, shading="gouraud", cmap="viridis"
)
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")
plt.title("Spektrogramm (Leistung) - Mikro 2")
plt.colorbar(pcm2_db, label="Spektrale Leistung [dB]")

# Rechts: Phase
plt.subplot(2, 2, 4)
pcm2_ph = plt.pcolormesh(
    t_spec2, f2, phase2, shading="gouraud", cmap="twilight"
)
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")
plt.title("Phasen-Spektrogramm - Mikro 2")
cbar2 = plt.colorbar(
    pcm2_ph,
    label="Phase [Radiant]",
    ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
)
cbar2.ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])

plt.tight_layout()
plt.show()

phase_diff = phase1 - phase2
plt.figure(figsize=(15, 5))
pcm_diff = plt.pcolormesh(
    t_spec1, f1, phase_diff, shading="gouraud", cmap="twilight"
)
plt.xlabel("Zeit [s]")
plt.ylabel("Frequenz [Hz]")
plt.title("Phasendifferenz - Mikrofon 1 - Mikrofon 2")
cbar_diff = plt.colorbar(
    pcm_diff,
    label="Phase [Radiant]",
    ticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
)
cbar_diff.ax.set_yticklabels(["-π", "-π/2", "0", "π/2", "π"])
plt.show()

#Mittelwert der Phasendifferenz
mean_phase_diff = np.mean(phase_diff, axis=1)  
#Zeitdifferenz berechnen
#c = 343  # Schallgeschwindigkeit in Luft in m/s
#d = 0.2  # Abstand zwischen den Mikrofonen in Metern
time_diff = mean_phase_diff / (2 * np.pi * f1)  # Zeitdifferenz in Sekunden
print(time_diff)
plt.figure(figsize=(15, 5))     
plt.plot(f1, mean_phase_diff, color="purple")
plt.xlabel("Frequenz [Hz]") 
plt.ylabel("Mittlere Phasendifferenz [Radiant]")
plt.title("Mittlere Phasendifferenz über die Zeit")
plt.grid()
plt.show()
# --- Methode 1: Phasendifferenz an der dominanten Frequenz ---

# 1. Finde den Index der maximalen Frequenz im Spektrum (sollte bei ca. 2000 Hz liegen)
# Wir suchen nur in der ersten Hälfte (positive Frequenzen)
idx_peak = np.argmax(FFT_mikro_1[:N//2])
freq_peak = freqs[idx_peak]

print(f"Dominante Frequenz gefunden bei: {freq_peak:.2f} Hz")

# 2. Kreuzspektrum NUR für diese Frequenz auswerten
x_peak = fft(mikro_1)[idx_peak]
y_peak = fft(mikro_2)[idx_peak]

phase_diff_peak = np.angle(x_peak * np.conj(y_peak))

print(f"Phasendifferenz bei {freq_peak:.2f} Hz: {phase_diff_peak:.4f} Radiant")

# 3. Zeitdifferenz und Winkel berechnen
c = 343.0  # Schallgeschwindigkeit in m/s
d = 0.08575    # Mikrofonabstand in m 

angle_rad=np.arcsin(phase_diff_peak/np.pi)
angle_deg=np.degrees(angle_rad)
print(f"Berechneter Einfallswinkel: {angle_deg:.2f} Grad")