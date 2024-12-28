import numpy as np
import scipy.io.wavfile as wav
from scipy.linalg import svd
import os

def noise_cancellation(input_file, output_file, threshold):
    # Loading audio file
    if not os.path.exists(input_file):
        print("File not found")
        return
    
    rate, data = wav.read(input_file)

    # Stereo to mono conversion
    if data.ndim > 1:
        data = data.sum(axis=1)
    
    # Normalizing audio data
    data = data.astype(np.float64)
    data /= np.max(np.abs(data))

    # Reshaping audio data into a Matrix
    chunk_size = 1024
    num_chunks = len(data) // chunk_size
    audio_matrix = np.reshape(data[:num_chunks * chunk_size], (num_chunks, chunk_size))

    # Applying SVD to the audio Matrix
    u, sigma, v_transposed = svd(audio_matrix, full_matrices=False)

    # Filtering out noise
    anti_sigma = np.where(sigma <= threshold, sigma, 0)
    filtered_sigma = sigma - anti_sigma

    # Reconstructing audio signal
    filtered_audio = np.dot(u, np.dot(np.diag(filtered_sigma), v_transposed)).flatten()

    # Denormalizing audio signal
    filtered_audio = (filtered_audio * 32767).astype(np.int16)

    # Saving audio file
    wav.write(output_file, rate, filtered_audio)
    print("Noise Cancellation Done")

input_file = input("Insert .wav file path: ")
output_file = input("Insert output file path: ")
threshold = float(input("Insert threshold: "))
noise_cancellation(input_file, output_file, threshold)