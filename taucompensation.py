import wave
import numpy as np
import matplotlib.pyplot as plt
import sys
import subprocess
np.set_printoptions(threshold=sys.maxsize)


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

audio_data_16bit = np.int16(Signal_A * 32767)
    
FRAME_RATE = 48000
with wave.open("Pilot_signal.wav", mode="wb") as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(FRAME_RATE)
    wav_file.writeframes(bytes(audio_data_16bit))
    wav_file.close()

print("Spiele Pilot-Signal ab...")
subprocess.run(["pw-play", "Pilot_signal.wav"])
print("Abspielen beendet.")