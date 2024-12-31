from ovito.io import import_file
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as k_B
from lammps_logfile import File
from consts import d_hat_i, ref_hairpin_positions

# Load trajectory
pipeline = import_file("../data/dump.atom.lammpstrj", multiple_frames=True)
# log = File("../data/log.lammps")


N = len(pipeline.compute(0).particles.positions)

differences_per_frame = []

n_frames = 100000

# T = np.array(log.get("Temp"))

for frame_index in range(n_frames):
    data = pipeline.compute(frame_index)

    particle_types = np.array(data.particles["Particle Type"])
    particle_ids = np.array(data.particles["Particle Identifier"])
    sorted_ids = np.argsort(particle_ids)
    sorted_positions = np.array(data.particles.positions)[sorted_ids]

    differences = []

    for particle_index in range(N // 2):
        # Particles are not sorted by id
        corresponding_particle_index = N - particle_index - 1

        pos = sorted_positions[particle_index]
        corresponding_pos = sorted_positions[corresponding_particle_index]

        differences.append(pos - corresponding_pos)

    differences_per_frame.append(differences)

differences_per_frame = np.array(differences_per_frame)

d_i = np.linalg.norm(differences_per_frame, axis=2)

rmsd = np.sqrt(np.sum((d_i - d_hat_i) ** 2, axis=1))

plt.plot(np.arange(len(rmsd)), rmsd)
plt.show()

num_bins = int(n_frames / 4)
density, bins = np.histogram(rmsd, density=True, bins=num_bins)
bin_widths = bins[1:] - bins[:-1]
unity_density = density / np.sum(density)

print(np.sum(unity_density))
bin_indices = np.digitize(rmsd, bins) - 2


T = T[::10]
P_s = unity_density[bin_indices]
F_s = k_B * T * np.log(P_s)

plt.scatter(rmsd_per_frame, F_s, s=0.1)
plt.show()
