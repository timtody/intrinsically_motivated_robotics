import gym
#import rlbench.gym
from models import FCPolicyCont
from a2c import A2CAgent
from utils import SkipWrapper
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import wandb
from ppo import PPO, Memory

cnf = OmegaConf.load("conf/conf.yaml")
cnf.merge_with_cli()

env = gym.make('LunarLander-v2')
# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')
# logging and hyperparameters
cnf = OmegaConf.load("conf/conf.yaml")
cnf.merge_with_cli()
# move to cnf file
obs_space = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = PPO(obs_space, action_dim, **cnf.ppo)
memory = Memory()

skip_wrapper = SkipWrapper(4)
env = skip_wrapper(env)
training_steps = int(1e6)

obs = env.reset()

episode_len = 0
timestep = 0
episode_reward = 0
rewards = []
wandb.init(name="mc-discrete", project="test")
for i in range(training_steps):

    done = False
    while not done:
        timestep += 1
        action = agent.policy_old.act(obs, memory)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        env.render()  # Note: rendering increases step time.
        if timestep % 100 == 0:
            # print("training")
            agent.update(memory)
            memory.clear_memory()
            timestep = 0
    print("Done! Last reward was", reward)
    wandb.log({"reward": episode_reward})
    episode_reward = 0
    obs = env.reset()
    done = False

print('Done')
env.close()
