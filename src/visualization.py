import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

def visualize_audio(file_path):
    """
    Membuat visualisasi audio berupa waveform dari file .wav

    """
    try:
        # Membaca file audio
        rate, data = read(file_path)
        if data.ndim > 1:
            data = data[:, 0]  # Mengambil satu channel jika stereo

        # Normalisasi data audio
        data = data.astype(np.float64) / np.max(np.abs(data))

        # Visualisasi Waveform
        plt.figure(figsize=(10, 4))
        plt.plot(data, color='blue')
        plt.title("Audio Waveform")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

audio_type = input("Audio type (input/noise/output): ")
if audio_type == "input":
    INPUT_FILES_PATH = "../data/audio-files/"
elif audio_type == "noise":
    INPUT_FILES_PATH = "../data/noise-files/"
elif audio_type == "output":
    INPUT_FILES_PATH = "../data/output-files/"

file_name = input("File name: ")
file_path = INPUT_FILES_PATH + file_name
visualize_audio(file_path)
