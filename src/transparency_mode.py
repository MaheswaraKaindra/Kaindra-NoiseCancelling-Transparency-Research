import numpy as np
from singular_value_decomposition import singular_value_decomposition as svd
import scipy.io.wavfile as wav

def transparency_mode(main_audio, noise_audio, output_path, noise_volume):
    """
    Implementasi mode transparansi pada audio file menggunakan SVD.
    """
    
    # Memuat file audio input dan noise
    main_rate, main_data = wav.read(main_audio)
    noise_rate, noise_data = wav.read(noise_audio)

    # Mengubah format audio stereo menjadi mono
    if main_data.ndim > 1:
        main_data = main_data.mean(axis=1)
    if noise_data.ndim > 1:
        noise_data = noise_data.mean(axis=1)

    # Menormalkan audio input dan noise
    main_data = main_data.astype(np.float64) / np.max(np.abs(main_data))
    noise_data = noise_data.astype(np.float64) / np.max(np.abs(noise_data))

    # Mengatur panjang noise agar sesuai dengan panjang audio input
    if len(noise_data) > len(main_data):
        noise_data = noise_data[:len(main_data)]
    elif len(noise_data) < len(main_data):
        noise_data = np.pad(noise_data, (0, len(main_data) - len(noise_data)), mode='constant', constant_values=0)

    # Merekonstruksi audio input dan noise dalam bentuk matriks
    chunk_size = 1024
    num_chunks = min(len(main_data), len(noise_data)) // chunk_size
    main_matrix = np.reshape(main_data[:num_chunks * chunk_size], (num_chunks, chunk_size))
    noise_matrix = np.reshape(noise_data[:num_chunks * chunk_size], (num_chunks, chunk_size))

    # Melakukan SVD pada matriks audio input dan noise
    main_U, main_Sigma, main_Vt = svd(main_matrix)
    noise_U, noise_Sigma, noise_Vt = svd(noise_matrix)

    # Menyamakan dimensi main_Sigma dan noise_Sigma
    Sigma_noise_matrix = np.zeros((noise_U.shape[1], noise_Vt.shape[0])) 
    np.fill_diagonal(Sigma_noise_matrix, noise_Sigma[:min(len(noise_Sigma), Sigma_noise_matrix.shape[0])])

    # Menyesuaikan volume noise
    Sigma_noise_matrix *= noise_volume

    # Menyamakan dimensi noise_U dan noise_Vt
    noise_U = noise_U[:, :main_U.shape[1]]
    noise_Vt = noise_Vt[:main_Vt.shape[0], :]

    # Merekonstruksi noise yang disesuaikan
    adjusted_noise_matrix = np.dot(noise_U, np.dot(Sigma_noise_matrix, noise_Vt))

    # Menggabungkan audio input dan noise
    combined_matrix = main_matrix + adjusted_noise_matrix

    # Merekonstruksi audio gabungan
    combined_audio = combined_matrix.flatten()
    combined_audio = (combined_audio * 32767).astype(np.int16)

    # Menyimpan audio output pada direktori yang ditentukan
    wav.write(output_path, main_rate, combined_audio)
    print("Transparency Mode Done")

# Input and output file paths
INPUT_FILES_PATH = "../data/audio-files/"
OUTPUT_FILES_PATH = "../data/output-files/"
NOISE_FILES_PATH = "../data/noise-files/"

temp_input_file = input("Insert .wav file path: ")
input_file = INPUT_FILES_PATH + temp_input_file
temp_noise_file = input("Insert noise file path: ")
noise_file = NOISE_FILES_PATH + temp_noise_file
temp_input_file = input("Insert output file path: ")
output_file = OUTPUT_FILES_PATH + temp_input_file
threshold = float(input("Insert threshold (0.5 is highly recommended): "))

transparency_mode(input_file, noise_file, output_file, threshold)
