import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Example: list of global points (x, y, z)
points = np.array([
    [0.0, 0.0, 2.0],
    [2.0, 2.0, 2.0],
    [4.0, 0.0, 2.0],
    [2.0, -2.0, 2.0],
    [0.0, 0.0, 2.0]
])

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linestyle='--', marker='o', color='blue')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Global Path Visualization')

# Optional: adjust view angle
ax.view_init(elev=30, azim=45)  # Try tweaking these numbers for a different 3D angle

plt.tight_layout()
plt.show()
