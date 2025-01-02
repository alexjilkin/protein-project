import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.data import DataCollection
from consts import d_hat_i
import zipfile


def extract_zip(zip_path: str, extract_to: str):
    """
    Extracts a zip file to the specified directory.
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def load_pipeline(file_path: str):
    """
    Loads a trajectory pipeline from a file.
    """
    return import_file(file_path, multiple_frames=True)


def compute_frame_distances(data: DataCollection, num_particles: int):
    """
    Computes pairwise distances between corresponding particles
    in the first and last halves of the positions array.
    """
    particle_ids = np.array(data.particles["Particle Identifier"])
    sorted_indices = np.argsort(particle_ids)
    sorted_positions = np.array(data.particles.positions)[sorted_indices]

    return np.linalg.norm(
        sorted_positions[: num_particles // 2]
        - sorted_positions[-1 : -(num_particles // 2) - 1 : -1],
        axis=1,
    )


def calculate_rmsd(d_i, d_hat_i):
    """
    Calculates the root-mean-square deviation (RMSD) given
    distances and reference distances.
    """
    return np.sqrt(np.sum((d_i - d_hat_i) ** 2, axis=1))


def calculate_free_energy(rmsd, k_B=1.0, T=0.02):
    """
    Calculates free energy based on RMSD and the given constants.
    """
    num_bins = int(np.sqrt(len(rmsd)))
    hist, bins = np.histogram(rmsd, density=True, bins=num_bins)

    P_s_bins = hist * np.diff(bins)
    P_s_bins[P_s_bins == 0] = 1e-10

    bin_indices = np.digitize(rmsd, bins, right=True) - 1
    P_s = P_s_bins[bin_indices]

    F_s = -k_B * T * np.log(P_s)
    F_s -= np.min(F_s)  # Normalize to start from zero

    sorted_indices = np.argsort(rmsd)
    return rmsd[sorted_indices], F_s[sorted_indices]


def plot_free_energy(rmsd, F_s):
    """
    Plots free energy versus RMSD.
    """
    plt.plot(rmsd, F_s)
    plt.xlabel("RMSD")
    plt.ylabel("F(s)")
    plt.title("Free Energy vs RMSD")
    plt.show()


def main():
    zip_path = "../data/dump.atom.lammpstrj.zip"
    extract_to = "../data/"
    trajectory_file = "../data/dump.atom.lammpstrj"

    extract_zip(zip_path, extract_to)
    pipeline = load_pipeline(trajectory_file)

    num_particles = len(pipeline.compute(0).particles.positions)

    d_i = []
    for frame_index in range(pipeline.source.num_frames):
        data = pipeline.compute(frame_index)
        d_i_per_frame = compute_frame_distances(data, num_particles)
        d_i.append(d_i_per_frame)

    d_i = np.array(d_i)
    rmsd = calculate_rmsd(d_i, d_hat_i)
    rmsd, F_s = calculate_free_energy(rmsd)

    plot_free_energy(rmsd, F_s)


if __name__ == "__main__":
    main()
