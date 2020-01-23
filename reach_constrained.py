import gym
import wandb
import rlbench.gym
from omegaconf import OmegaConf
from ppo_cont import PPO, Memory
from wrappers import ObsWrapper
from imageio import get_writer
from logger import Logger
import numpy as np

import torch


# logging and hyperparameters
cnf = OmegaConf.load("conf/constrained.yaml")
cnf.merge_with_cli()
OmegaConf.set_struct(cnf, True)
#Logger.setup(cnf)

env = gym.make(cnf.main.env_name)
env = ObsWrapper(env)

obs_space = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(obs_space, action_dim, **cnf.ppo)
memory = Memory()

def get_target():
    return env.task._task.target.get_position()

def get_tip():
   return env.task._task.robot.arm.get_tip().get_position() 

def get_distance():
    target = torch.tensor(get_target())
    tip = torch.tensor(get_tip())
    return torch.dist(target, tip)

obs = env.reset()
done = False

for i in range(1000):
    while not done:
        obs, reward, done, info = env.step(env.action_space.sample())
        print(get_distance().item())
        env.render()