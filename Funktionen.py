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

def plot_polar_spectrum(phi_deg, P_values_dB, final_angle, freq_peak):
    """
    Erzeugt einen Halbkreis-Polarplot (0° Rechts, 90° Oben, 180° Links).
    
    Parameter:
    - phi_deg: Array von Winkeln in Grad (muss von 0 bis 180 gehen)
    - P_values_dB: Array der berechneten Leistungen in dB
    - final_angle: Der berechnete Haupt-Einfallswinkel in Grad
    - freq_peak: Die Frequenz, bei der gemessen wurde
    """
    phi_rad = np.radians(phi_deg)

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 5))
    ax.plot(phi_rad, P_values_dB, linewidth=2, color='#1f77b4')

    # --- Das Koordinatensystem einstellen (0° Rechts, 180° Links) ---
    ax.set_theta_zero_location('E')  # 'E' steht für East (Rechts) -> 0° ist hier
    ax.set_theta_direction(1)        # 1 = Gegen den Uhrzeigersinn (Standard Mathe-Konvention)
    ax.set_thetamin(0)               # Halbkreis beginnt rechts bei 0°
    ax.set_thetamax(180)             # Halbkreis endet links bei 180°

    # Y-Achse (dB) skalieren
    y_min = max(-15, np.min(P_values_dB)) 
    ax.set_ylim([y_min, 0])
    
    # Die dB-Beschriftungen (die Ringe) auf die 90°-Achse (Mitte oben) legen, 
    # damit sie rechts und links nicht mit der 0°-180°-Linie kollidieren
    ax.set_rlabel_position(90) 

    # Titel setzen
    plt.title(f"DOA Bartlett Spektrum bei {freq_peak:.0f} Hz\nGefundener Winkel: {final_angle}°\n", va='bottom')
    
    plt.tight_layout()
    plt.show()