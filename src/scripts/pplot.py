import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Example: list of global points (x, y, z)
# points = np.array([
#     [0.0, 0.0, 2.0],
#     [2.0, 2.0, 2.0],
#     [4.0, 0.0, 2.0],
#     [2.0, -2.0, 2.0],
#     [0.0, 0.0, 2.0]
# ])


points = np.array([(2, 2, 0.5),
(-0.05, 1.24, 0.5),
(-1.30, 0.036, 0.5),
(-1.84, -1.71, 0.5)])
points = [(0.00012787370360456407, -0.00011228654329897836, 0.00012751005124300718), (0.00011144268501084298, -0.45012316276843195, 5.8919376897392794e-05), (0.40024432127247567, -0.6501059510686901, 0.00027407536981627345), (0.8000974470603979, -0.8500307552836603, 0.00032004000968299806), (0.8303051620721817, -0.8162282392382623, 0.7386496067047119), (0.8782464921474458, -1.0297814078629017, 0.9889940619468689), (1.166852355003357, -0.8290716528892517, 0.9661797881126404), (2.6011533737182617, -1.2906063795089722, 0.9212203025817871), (3.4363157749176025, -1.9986157417297363, 0.9394797682762146), (4.106590270996094, -3.2116315364837646, 1.0339025259017944), (4.3079047203063965, -3.967740297317505, 1.055363416671753), (4.513303756713867, -4.518580913543701, 1.090603232383728)]
points = np.array(points)

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, linestyle='--', marker='o', color='blue')

test = [(1.74855479e-04,-3.76146381e-05 , 6.60237638e-05),
        [ 2.14283466, -0.96262795,  0.93171269]]
test = np.array(test)
x = test[:, 0]
y = test[:, 1]
z = test[:, 2]
ax.plot(x, y, z, linestyle='--', marker='o', color='red')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Global Path Visualization')

# Optional: adjust view angle
ax.view_init(elev=30, azim=45)  # Try tweaking these numbers for a different 3D angle

plt.tight_layout()
plt.show()
