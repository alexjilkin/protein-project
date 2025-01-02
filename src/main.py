from ovito.io import import_file
import numpy as np
import matplotlib.pyplot as plt
from consts import d_hat_i
import zipfile

with zipfile.ZipFile("../data/dump.atom.lammpstrj.zip", "r") as zip_ref:
    zip_ref.extractall("../data/")

pipeline = import_file("../data/dump.atom.lammpstrj", multiple_frames=True)

N = len(pipeline.compute(0).particles.positions)

d_i = []
for frame_index in range(pipeline.source.num_frames):
    data = pipeline.compute(frame_index)

    particle_types = np.array(data.particles["Particle Type"])
    particle_ids = np.array(data.particles["Particle Identifier"])

    # Order trajectory positions by id
    sorted_ids = np.argsort(particle_ids)
    sorted_positions = np.array(data.particles.positions)[sorted_ids]

    # Calculate distance between corresponding particles
    d_i_per_frame = np.linalg.norm(
        sorted_positions[: N // 2] - sorted_positions[-1 : -(N // 2) - 1 : -1], axis=1
    )

    d_i.append(d_i_per_frame)

rmsd = np.sqrt(np.sum((d_i - d_hat_i) ** 2, axis=1))

num_bins = int(np.sqrt(len(rmsd)))
hist, bins = np.histogram(rmsd, density=True, bins=num_bins)

# Normalise and handle 0
P_s_bins = hist * np.diff(bins)
P_s_bins[P_s_bins == 0] = 1e-10

# Find the bin indices, right=True to align with np.histogram behaviour
bin_indices = np.digitize(rmsd, bins, right=True) - 1
P_s = P_s_bins[bin_indices]

# k_B is already inclued in Lennard-Jones, T constant temperature of Langevin thermo
k_B, T = 1.0, 0.02

F_s = -k_B * T * np.log(P_s)

# Sort for plotting
sorted_indices = np.argsort(rmsd)
rmsd = rmsd[sorted_indices]
F_s = F_s[sorted_indices]
F_s -= np.min(F_s)

plt.plot(rmsd, F_s)
plt.xlabel("RMSD")
plt.ylabel("F(s)")
plt.show()
