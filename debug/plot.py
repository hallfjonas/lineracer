
from lineracer import *

track = RaceTrack.generate_random_track()
limits_x, limits_y = track.get_limits()
x = np.linspace(limits_x[0], limits_x[1], 100)
y = np.linspace(limits_y[0], limits_y[1], 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)
O = np.zeros_like(X)
for i in range(Z.shape[0]):
    for j in range(Z.shape[1]):
        Z[i,j] = track.distance_to_middle_line([X[i,j], Y[i,j]])
        O[i,j] = track.is_on_track([X[i,j], Y[i,j]])

print(f"Track width: {track.width}")

fig, ax = plt.subplots()
track.plot_track(ax=ax)
CS = ax.contour(X, Y, Z)
ax.clabel(CS, fontsize=10)
plt.show()

fig, ax = plt.subplots()
ax.contour(X, Y, O)
plt.show()

