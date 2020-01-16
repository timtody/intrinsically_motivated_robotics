from rlbench.environment import Environment
from rlbench.action_modes import ActionMode, ArmActionMode
from rlbench.tasks import ReachTarget
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from models import Sphere
import numpy as np
import wandb
import pickle


fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(1, 3)

action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
env = Environment(action_mode)
env.launch()
# wandb.init(project="test", name="curaudo")

task = env.get_task(ReachTarget)
descriptions, obs = task.reset()
positions = []

# adjust joint for maximum length
for i in range(18):
    obs, reward, terminate = task.step([0, 0, 0, 0, 0, 1, 0, 0])

# adjust second joint
for i in range(15):
    obs, reward, terminate = task.step([0, 0, 0, 1, 0, 0, 0, 0])

# make the hull

for i in range(40):
    obs, reward, terminate = task.step([0, 1, 0, 0, 0, 0, 0, 0])
    gripper_position = env._robot.arm.get_tip().get_position()[:3]
    positions.append(gripper_position)

for i in range(80):
    obs, reward, terminate = task.step([0, -1, 0, 0, 0, 0, 0, 0])
    gripper_position = env._robot.arm.get_tip().get_position()[:3]
    positions.append(gripper_position)

for i in range(30):
    obs, reward, terminate = task.step([-1, 0, 0, 0, 0, 0, 0, 0])
    gripper_position = env._robot.arm.get_tip().get_position()[:3]
    positions.append(gripper_position)

for i in range(80):
    obs, reward, terminate = task.step([0, 1, 0, 0, 0, 0, 0, 0])
    gripper_position = env._robot.arm.get_tip().get_position()[:3]
    positions.append(gripper_position)

# examinate
for i in range(60):
    obs, reward, terminate = task.step([0, 0, 0, 0, 0, 0, 0, 0])
    gripper_position = env._robot.arm.get_tip().get_position()[:3]
    positions.append(gripper_position)


# estimate sphere
sphere_model = Sphere()
radius, origin, losses = sphere_model.train(positions, 20)
print(losses)
print(radius, origin)

pickle.dump(positions, open("positions.p", "wb"))
"""
for i in range(80):
    obs, reward, terminate = task.step([0, 1, 0, 0, 0, 0, 0, 0])
    gripper_position = env._robot.arm.get_tip().get_position()[:3]
    positions.append(gripper_position)
    print(gripper_position)
    print(env._robot.arm.get_position())

wandb.log({
            "gripper positions": wandb.Object3D(np.array(positions))
            })
"""
plt.plot(*zip(*positions))
plt.show()

env.shutdown()
