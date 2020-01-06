import gym
import rlbench.gym
from matplotlib import pyplot as plt

# env = gym.make('reach_target-state-v0')
# Alternatively, for vision:
env = gym.make('reach_target-vision-v0')
training_steps = 120
episode_length = 40
for i in range(training_steps):
    if i % episode_length == 0:
        print('Reset Episode')
        obs = env.reset()
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    exit(1)
    #env.render()  # Note: rendering increases step time.

print('Done')
env.close()