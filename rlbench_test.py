import gym
import rlbench.gym
from pyrep.objects.shape import Shape
from utils import Plotter3D
import numpy as np
import plotly.graph_objects as go


# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')

env = gym.make("close_drawer-state-v0")

training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    env.render()  # Note: rendering increases step time.

print('Done')
env.close()
"""

training_steps = 40
obs = env.reset()
table = Shape("diningTable")
bbox_table = table.get_model_bounding_box()
x0 = np.linspace(bbox_table[0], bbox_table[1], 20)
y0 = np.linspace(bbox_table[2], bbox_table[3], 20)
z0 = np.linspace(bbox_table[4], bbox_table[5], 20)

x = np.outer(np.outer(x0, y0), z0).flatten()
y = np.outer(np.outer(z0, x0), y0).flatten()
z = np.outer(np.outer(y0, z0), x0).flatten()


fig = go.Figure(data=[
    go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=12,
            color="brown"
        )

    )
])
fig.show()
exit(1)
for i in range(training_steps):
    obs, reward, lterminate, _ = env.step(env.action_space.sample())
    print(reward)
    env.render(mode="rgb_array")  # Note: rendering increases step time.
    print("test")
print('Done')
env.close()
"""