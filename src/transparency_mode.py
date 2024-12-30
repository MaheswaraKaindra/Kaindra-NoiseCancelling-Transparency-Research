import numpy as np
from singular_value_decomposition import singular_value_decomposition as svd
import scipy.io.wavfile as wav

def transparency_mode(main_audio, background_audio, output_path, rain_volume=0.5):
    rate_main, data_main = wav.read(main_audio)
    rate_bg, data_bg = wav.read(background_audio)

    assert rate_main == rate_bg, "Sample rates must match!"

    if data_main.ndim > 1:
        data_main = data_main.mean(axis=1)
    if data_bg.ndim > 1:
        data_bg = data_bg.mean(axis=1)

    data_main = data_main.astype(np.float64) / np.max(np.abs(data_main))
    data_bg = data_bg.astype(np.float64) / np.max(np.abs(data_bg))

    if len(data_bg) > len(data_main):
        data_bg = data_bg[:len(data_main)]
    elif len(data_bg) < len(data_main):
        data_bg = np.pad(data_bg, (0, len(data_main) - len(data_bg)), mode='constant', constant_values=0)

    chunk_size = 1024
    num_chunks_main = len(data_main) // chunk_size
    num_chunks_bg = len(data_bg) // chunk_size

    main_matrix = np.reshape(data_main[:num_chunks_main * chunk_size], (num_chunks_main, chunk_size))
    bg_matrix = np.reshape(data_bg[:num_chunks_bg * chunk_size], (num_chunks_bg, chunk_size))

    U_main, Sigma_main, Vt_main = svd(main_matrix)
    U_bg, Sigma_bg, Vt_bg = svd(bg_matrix)

    Sigma_bg *= rain_volume

    adjusted_bg_matrix = np.dot(U_bg, np.dot(np.diag(Sigma_bg), Vt_bg))

    combined_matrix = main_matrix + adjusted_bg_matrix[:main_matrix.shape[0], :]

    combined_audio = combined_matrix.flatten()
    combined_audio = (combined_audio * 32767).astype(np.int16)

    wav.write(output_path, rate_main, combined_audio)
