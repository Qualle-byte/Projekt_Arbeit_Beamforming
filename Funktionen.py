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


def Zeitbereich_Plot(t,mikro1, mikro2, mikro3, mikro4):
    # Plot für Mikrofon 1
    plt.subplot(2, 2, 1)
    plt.plot(t, mikro1, color="blue")
    plt.title("Mikrofon 1 (Linker Kanal)")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    # Plot für Mikrofon 2
    plt.subplot(2, 2, 2)
    plt.plot(t, mikro2, color="orange")
    plt.title("Mikrofon 2 (Rechter Kanal)")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    # Plot für Mikrofon 3
    plt.subplot(2, 2, 3)
    plt.plot(t, mikro3, color="green")
    plt.title("Mikrofon 3 (Dritter Kanal)")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    # Plot für Mikrofon 4
    plt.subplot(2, 2, 4)
    plt.plot(t, mikro4, color="red")
    plt.title("Mikrofon 4 (Vierter Kanal)")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.tight_layout()
    plt.show()
    return 0

def Zeitbereich_alle_in_einem_Plot(t,mikro1, mikro2, mikro3, mikro4):
    plt.figure(figsize=(10, 6))
    plt.plot(t, mikro1, label="Mikrofon 1 (Linker Kanal)", color="blue")
    plt.plot(t, mikro2, label="Mikrofon 2 (Rechter Kanal)", color="orange")
    plt.plot(t, mikro3, label="Mikrofon 3 (Dritter Kanal)", color="green")
    plt.plot(t, mikro4, label="Mikrofon 4 (Vierter Kanal)", color="red")
    
    plt.title("Zeitbereichsanalyse aller Mikrofone")
    plt.xlabel("Zeit [s]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    return 0

def normiertefunktionen(mikro_daten):
    normiert_mikros = mikro_daten / np.linalg.norm(mikro_daten)
    return normiert_mikros

def Autokovarianzmatrix(audiosignal, N):
    mikro_daten = audiosignal[:, 0:4]
    mikro_daten_conj = mikro_daten.conj()

    # Autokovarianzmatrix
    R_xx= np.zeros((4,4))
    for s in range(4):
        for z in range(4):
            for i in range(N):
                    R_xx[s][z] = (mikro_daten[i][s] * mikro_daten_conj[i][z])/N
    #print("Das ist die händisch ausgerechnete Autokovarianzmatrix R_xx:")
    #print(R_xx)
    return R_xx

def Autokovarianzmatrix_schneller(audio_signal, N):
    mikro_daten = audio_signal[:, 0:4]
    mikro_daten_conj = mikro_daten.conj()

    # Autokovarianzmatrix
    R_xx= np.zeros((4,4))
    for s in range(4):
        for z in range(4):
                    R_xx[s][z] = np.dot(audio_signal[:,s], audio_signal[:,z].conj())/N
    #print("Das ist die händisch ausgerechnete Autokovarianzmatrix R_xx:")
    #print(R_xx)
    return R_xx


def Winkel_LDS(x_peak, y_peak,freq_peak,d):
     c = 343.0
     phase_diff_peak = np.angle(x_peak * np.conj(y_peak))
     angle_rad=np.arcsin((phase_diff_peak*c)/(2*np.pi*freq_peak*d))
     angle_deg=np.degrees(angle_rad)
     print(f"Berechneter Einfallswinkel mit Kreuzleistungsdichtespektrum: {angle_deg:.2f} °")
     return 0