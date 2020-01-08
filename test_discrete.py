import gym
from models import FCPolicy
from a2c import A2CAgent
from ppo import PPO, Memory
from utils import SkipWrapper
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import wandb
env = gym.make('CartPole-v0')

obs_space = env.observation_space.shape[0]
n_actions = env.action_space.n
n_latent_var = 64
lr = 0.002
betas = (0.9, 0.999)
gamma = 0.99                # discount factor
K_epochs = 4                # update policy for K epochs
eps_clip = 0.2              # clip param for ppo

agent = PPO(obs_space, n_actions, n_latent_var, lr, 
            betas, gamma, K_epochs, eps_clip)
memory = Memory()
training_steps = int(1e6)

obs, done = env.reset(), False

episode_len = 0
episode_reward = 0
rewards = []
timestep = 0
wandb.init(name="ppo-test", project="curious")
for i in range(training_steps):
    timestep += 1
    if done:
        wandb.log({"reward": episode_reward})
        rewards.append(episode_reward)
        episode_reward = 0
        obs = env.reset()
    action = agent.policy_old.act(obs, memory)
    obs, reward, done, _ = env.step(action)
    episode_reward += reward
    memory.rewards.append(reward)
    memory.is_terminals.append(done)
    if timestep % 1000 == 0:
        agent.update(memory)
        memory.clear_memory()
        timestep = 0
    env.render()  # Note: rendering increases step time.
    
print('Done')
env.close()