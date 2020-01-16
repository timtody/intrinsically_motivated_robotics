import gym
import rlbench.gym
import numpy as np
from wrappers import ObsWrapper
import wandb

env = gym.make('reach_target-state-v0')
env = ObsWrapper(env)
# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')

training_steps = 1000
episode_length = 40
direction = True
gripper_positions = []
wandb.init(project="test", name="curaudo")

for i in range(training_steps):
    if i % episode_length == 0:
        direction = not direction
        print('Reset Episode')
        obs = env.reset()
    state, reward, terminate, _ = env.step([np.random.choice([0, 1]), np.random.choice([0, 1]), 0, 0, 0, 0, 0, 0])
    gripper_positions.append(np.array([*state.gripper_pose[:3],
                                       128, 128, 128]
                                      )
                             )
    # env.render()

wandb.log({
            "gripper positions": wandb.Object3D(np.array(gripper_positions))
            })

print('Done')
env.close()
