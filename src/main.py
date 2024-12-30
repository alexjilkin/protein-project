from ovito.io import import_file
from matplotlib.pyplot import plt

# Load trajectory
pipeline = import_file("dump.atom.lammpstrj", multiple_frames=True)

data = pipeline.compute()
positions = data.particles.positions
print(positions[0])
# plt.plot(range(len(positions)), positions)
