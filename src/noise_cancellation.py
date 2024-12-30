import numpy as np
import scipy.io.wavfile as wav
from singular_value_decomposition import singular_value_decomposition as svd
import os

INPUT_FILES_PATH = "../data/audio-files/"
OUTPUT_FILES_PATH = "../data/output-files/"
def noise_cancellation(input_file, output_file, threshold):
    """
    Implementasi noise cancellation pada audio file menggunakan SVD.
    """

    # Memuat file audio
    if not os.path.exists(input_file):
        print("File not found")
        return
    
    rate, data = wav.read(input_file)

    # Mengonversi audio stereo menjadi mono
    if data.ndim > 1:
        data = data.sum(axis=1)
    
    # Normalisasi sinyal audio
    data = data.astype(np.float64)
    data /= np.max(np.abs(data))

    # Membentuk sinyal audio dalam bentuk matriks (agar bisa dilakukan SVD)
    chunk_size = 1024
    num_chunks = len(data) // chunk_size
    audio_matrix = np.reshape(data[:num_chunks * chunk_size], (num_chunks, chunk_size))

    # Melakukan SVD pada matriks audio
    u, sigma, v_transposed = svd(audio_matrix, full_matrices=False)

    # Mengeluarkan noise dari sigma
    anti_sigma = np.where(sigma < threshold, sigma, 0)
    filtered_sigma = sigma - anti_sigma

    # Rekonstruksi audio
    filtered_audio = np.dot(u, np.dot(np.diag(filtered_sigma), v_transposed)).flatten()

    # Mengembalikan audio ke rentang aslinya
    filtered_audio = (filtered_audio * 32767).astype(np.int16)

    # Menyimpan audio hasil noise cancellation
    wav.write(output_file, rate, filtered_audio)
    print("Noise Cancellation Done")

temp_input_file = input("Insert .wav file path: ")
input_file = INPUT_FILES_PATH + temp_input_file
temp_input_file = input("Insert output file path: ")
output_file = OUTPUT_FILES_PATH + temp_input_file
threshold = int(input("Insert threshold (20 is highly recommended): "))
noise_cancellation(input_file, output_file, threshold)