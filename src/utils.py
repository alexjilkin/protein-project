import zipfile
import numpy as np
import pandas as pd

# Frame 111067
ref_hairpin_positions = np.array(
    [
        [-1.28254, -2.59199, 0.0376731],
        [-0.401262, -2.06241, -0.0779148],
        [0.508574, -1.64908, 0.020692],
        [1.34964, -1.13541, -0.0712788],
        [2.27988, -0.770173, -0.0688153],
        [3.12565, -0.264099, 0.0974064],
        [3.71572, 0.514255, -0.0699172],
        [2.92059, 1.10976, -0.00466215],
        [2.25606, 0.364234, 0.0319749],
        [1.31095, -0.0348316, -0.0256341],
        [0.42227, -0.515715, 0.0362422],
        [-0.441136, -0.998823, 0.0995269],
        [-1.37016, -1.44781, -0.0769311],
    ]
)
k = len(ref_hairpin_positions) // 2

d_hat_i = np.linalg.norm(
    ref_hairpin_positions[:k] - ref_hairpin_positions[-1 : -k - 1 : -1], axis=1
)


def extract_zip(zip_path: str, extract_to: str):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def save_descriptors_to_file(
    d_i: np.ndarray, d_hat_i: np.ndarray, s: np.ndarray, filename: str
):
    """
    Saves the descriptors (d_i, d_hat_i, RMSD) by rows of time step.
    """
    # Create column names dynamically for each d_i and d_hat_i descriptor
    num_components = d_i.shape[1]
    d_i_columns = [f"d_{i+1}" for i in range(num_components)]
    d_hat_i_columns = [f"d_hat_{i+1}" for i in range(num_components)]

    data = pd.DataFrame(
        np.hstack([d_i, np.tile(d_hat_i, (d_i.shape[0], 1)), s[:, np.newaxis]]),
        columns=d_i_columns + d_hat_i_columns + ["RMSD"],
    )

    data.to_csv(filename, index=False)
    print(f"Descriptors saved to {filename}")
