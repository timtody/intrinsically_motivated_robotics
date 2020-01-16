import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
from models import Sphere

import plotly.graph_objects as go
import wandb

#fig = plt.figure()
#ax = fig.add_subplot(121, projection="3d")
#ax2d = fig.add_subplot(122)
#ax.set_xlim(-1, 1)
#ax.set_ylim(-1, 1)
#.set_zlim(0, 2)
wandb.init(project="test")

radius = 0.9285090706636693
origin = [-0.2678461927495279, -0.006510627535040618, 1.0827146125969983]

theta = np.linspace(0, np.pi, 20)
phi = np.linspace(0, 2*np.pi, 20)

x0 = origin[0] + radius * np.outer(np.sin(theta), np.cos(phi)).flatten()
y0 = origin[1] + radius * np.outer(np.sin(theta), np.sin(phi)).flatten()
z0 = origin[2] + radius * np.outer(np.cos(theta), np.ones_like(theta)).flatten()

x = []
y = []
z = []

for xx, yy, zz in zip(x0, y0, z0):
    if zz > 0.85:
        x.append(xx)
        y.append(yy)
        z.append(zz)


positions = pickle.load(open("positions.p", "rb"))
px, py, pz = zip(*positions)
fig = go.Figure(data=[
    go.Scatter3d(x=x, y=y, z=z,
                 mode='markers'),
    go.Scatter3d(x=px, y=py, z=pz)
])
wandb.log({"plot": fig})
fig.show()

exit(1)

sphere_model = Sphere()
radius, origin, losses = sphere_model.train(positions, 1)
ax2d.plot(losses)


def sample_spherical(npoints, radius=1, origin=[0, 0, 1], ndim=3,
                     treshold=0.85):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    vec *= radius
    vec = vec.T
    vec += origin
    results = []
    for i, elem in enumerate(vec):
        if elem[2] > treshold:
            results.append(elem)
    # vec = vec.T
    results = zip(*results)
    return results


x, y, z = sample_spherical(100, radius, origin)

points = np.random.randint(0, high=len(positions), size=3)
for point in points:
    print(positions[point])


print(f"origin: {origin}\nradius: {radius}\nfinal loss: {losses[-1]}")
ax.scatter(x, y, z, c='r', zorder=10)
ax.plot(*zip(*positions))
plt.show()
