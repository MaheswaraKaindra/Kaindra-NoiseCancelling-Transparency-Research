import numpy as np

def singular_value_decomposition(A, full_matrices=True):
    """
    Implementasi SVD dengan cara 1 berdasarkan Munir (2024) dan scipy.linalg.svd.

    Cara kerja SVD terinspirasi dari scipy.linalg.svd.
    """
    # Hitung A^T A dan nilai eigen untuk mendapatkan V
    ATA = np.dot(A.T, A)
    eigen_values_V, V = np.linalg.eigh(ATA)

    # Urutkan nilai eigen dan vektor eigen dari besar ke kecil
    sorted_indices = np.argsort(eigen_values_V)[::-1]
    eigen_values_V = eigen_values_V[sorted_indices]
    V = V[:, sorted_indices]

    # Hitung nilai singular (akar nilai eigen)
    singular_values = np.sqrt(eigen_values_V)

    # Hitung U menggunakan A V / Sigma
    U = np.zeros((A.shape[0], len(singular_values)))
    for i in range(len(singular_values)):
        if singular_values[i] > 1e-10:  # Hindari pembagian dengan nol
            U[:, i] = np.dot(A, V[:, i]) / singular_values[i]

    # Normalisasi U
    U = U / np.linalg.norm(U, axis=0)

    # Memperluas U dan V jika full_matrices=True
    if full_matrices:
        m, n = A.shape
        if m > n:
            extra_cols = np.eye(m)[:, n:]
            U = np.hstack((U, extra_cols))
        elif n > m:
            extra_cols = np.eye(n)[m:, :]
            V = np.hstack((V, extra_cols.T))

    # Transpose V untuk menghasilkan V^T
    Vt = V.T

    return U, singular_values, Vt
