from ovito.io import import_file
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann as k_B
from lammps_logfile import File

# Load trajectory
pipeline = import_file("../data/dump.atom.lammpstrj", multiple_frames=True)
log = File("../data/log.lammps")

# Get the number of frames

N = len(pipeline.compute(0).particles.positions)
# Store results
squared_differences_per_frame = []

n_frames = pipeline.source.num_frames

T = np.array(log.get("Temp"))

for frame_index in range(n_frames):
    # Compute the data for the current frame
    data = pipeline.compute(frame_index)

    # Extract positions for all particles
    positions = data.particles.positions

    frame_squared_differences = []

    for particle_index in range(N // 2):
        pos = np.array(positions[particle_index])
        corresponding_pos = np.array(positions[N - particle_index - 1])

        # Calculate squared difference
        squared_diff = (pos - corresponding_pos) ** 2
        frame_squared_differences.append(squared_diff)

    squared_differences_per_frame.append(frame_squared_differences)

squared_differences_per_frame = np.array(squared_differences_per_frame)

# Summing x_i, y_i, z_i
total_squared_distances = np.sum(squared_differences_per_frame, axis=2)

# Taking the mean
mean_squared_distances = np.mean(total_squared_distances, axis=1)
rmsd_per_frame = np.sqrt(mean_squared_distances)


num_bins = int(n_frames / 4)
vals, bins = np.histogram(rmsd_per_frame, density=True, bins=num_bins)
bin_indices = np.digitize(rmsd_per_frame, bins) - 2

print(bin_indices)
print(len(vals))
print(vals)
# print(bin_indices)
# for ind in bin_indices:
#     print(ind)
#     print(vals[ind])

T = T[::10]
P_s = vals[bin_indices]
print(len(T))
F_s = k_B * T * np.log(P_s)
plt.scatter(rmsd_per_frame, F_s)

plt.show()
