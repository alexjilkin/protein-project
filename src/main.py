import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.data import DataCollection
from utils import d_hat_i, extract_zip


def compute_frame_distances(data: DataCollection):
    """
    Computes pairwise distances between corresponding particles
    in the first and last halves of the positions array. Sorts by
    Particle ID.
    """
    k = len(data.particles.positions)
    particle_ids = np.array(data.particles["Particle Identifier"])
    sorted_indices = np.argsort(particle_ids)
    sorted_positions = np.array(data.particles.positions)[sorted_indices]

    return np.linalg.norm(
        sorted_positions[: k // 2] - sorted_positions[-1 : -(k // 2) - 1 : -1],
        axis=1,
    )


def rmsd(d_i: np.ndarray, d_hat_i: np.ndarray):
    return np.sqrt(np.sum((d_i - d_hat_i) ** 2, axis=1))


def F(s: np.ndarray, k_B: float, T: float):
    """
    Computes the free energy surface by collective variable s.
    """
    num_bins = int(np.sqrt(len(s)))
    hist, bins = np.histogram(s, density=True, bins=num_bins)

    P_s_bins = hist * np.diff(bins)
    P_s_bins[P_s_bins == 0] = 1e-10

    # Get bin indices for s
    bin_indices = np.digitize(s, bins, right=True) - 1
    P_s = P_s_bins[bin_indices]

    F_s = -k_B * T * np.log(P_s)
    F_s -= np.min(F_s)  # Normalize to start from zero

    # Sorts by ascending rmsd
    sorted_indices = np.argsort(s)
    return s[sorted_indices], F_s[sorted_indices]


def main():
    zip_path = "../data/dump.atom.lammpstrj.zip"
    extract_to = "../data/"
    trajectory_file = "../data/dump.atom.lammpstrj"

    extract_zip(zip_path, extract_to)
    pipeline = import_file(trajectory_file, multiple_frames=True)

    d_i = np.array(
        [
            compute_frame_distances(pipeline.compute(frame_index))
            for frame_index in range(pipeline.source.num_frames)
        ]
    )

    # Generate descriptor. Can be saved to a file, using pickle for instance.
    s = rmsd(d_i, d_hat_i)

    # k_B=1 in Lennard-Jones framework
    # T=0.02 from Langevin constant thermo
    plt.plot(*F(s, 1.0, 0.02))
    plt.xlabel("RMSD")
    plt.ylabel("F(s)")
    plt.title("Toy Model Protein")
    plt.show()


if __name__ == "__main__":
    main()
