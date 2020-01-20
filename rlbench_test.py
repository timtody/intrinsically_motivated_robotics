import gym
import rlbench.gym
import time

env = gym.make('close_fridge-state-v0')
# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')

training_steps = 40
obs = env.reset()
begin = time.time()
for i in range(training_steps):
    obs, reward, terminate, _ = env.step(env.action_space.sample())
    print(reward)
    env.render(mode="rgb_array")  # Note: rendering increases step time.
    print("test")
end = time.time()
print("took", end - begin, "seconds.")
print('Done')
env.close()
