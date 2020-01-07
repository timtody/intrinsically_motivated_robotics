import gym
#import rlbench.gym
from models import FCPolicy
from a2c import A2CAgent
from utils import SkipWrapper
from matplotlib import pyplot as plt


env = gym.make('MountainCarContinuous-v0')
# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')
obs_space = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

fc_policy = FCPolicy(obs_space, action_dim)
agent = A2CAgent()
agent.set_policy_network(fc_policy)
skip_wrapper = SkipWrapper(4)
env = skip_wrapper(env)
training_steps = int(1e4)

obs, done = env.reset(), False

episode_len = 0
episode_reward = 0
rewards = []
for i in range(training_steps):
    if done:
        rewards.append(episode_reward)
        agent.train()
        episode_reward = 0
        episode_len = 0
        obs = env.reset()
    action = agent.get_action(obs)
    obs, reward, done, _ = env.step(env.action_space.sample())
    episode_len += 1 
    episode_reward += reward
    agent.append_reward(reward)
    agent.set_done(done)
    env.render()  # Note: rendering increases step time.
    
plt.plot(rewards)
print('Done')
env.close()