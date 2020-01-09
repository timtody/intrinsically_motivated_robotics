import gym
#import rlbench.gym
from models import FCPolicyCont
from a2c import A2CAgent
from utils import SkipWrapper
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import wandb

cnf = OmegaConf.load("conf/conf.yaml")
cnf.merge_with_cli()

env = gym.make('MountainCarContinuous-v0')
# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')
obs_space = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

fc_policy = FCPolicyCont(obs_space, action_dim)
agent = A2CAgent()
agent.set_policy_network(fc_policy)
skip_wrapper = SkipWrapper(2)
env = skip_wrapper(env)
training_steps = int(1e6)

obs, done = env.reset(), False

episode_len = 0
episode_reward = 0
rewards = []
wandb.init(name="mc-cont", project="curious")
for i in range(training_steps):
    if done:
        wandb.log({"reward": episode_reward})
        rewards.append(episode_reward)
        agent.train()
        episode_reward = 0
        obs = env.reset()
    action = agent.get_action_cont(obs)
    obs, reward, done, _ = env.step(action)
    episode_reward += reward
    agent.append_reward(reward)
    agent.set_done(done)
    env.render()  # Note: rendering increases step time.

print('Done')
env.close()