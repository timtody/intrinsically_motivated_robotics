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

if cnf.wandb.use:
    wandb.init(project=cnf.wandb.project,
               name=cnf.wandb.name,
               config=cnf)
    wandb.watch(agent.policy, log="all")


def get_target():
    return env.task._task.target.get_position()


def get_tip():
    return env.task._task.robot.arm.get_tip().get_position()


def get_distance():
    target = torch.tensor(get_target())
    tip = torch.tensor(get_tip())
    return torch.dist(target, tip)


MAX_LEN = 150
timestep = 0

for i in range(1000000):
    timestep += 1
    obs = env.reset()
    done = False
    episode_reward = 0
    episode_length = 0

    while not done:
        action = agent.policy_old.act(obs.get_low_dim_data(), memory)
        obs, reward, done, info = env.step(action)
        solved = done
        reward = -get_distance() + reward
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        if timestep % 100 == 0:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0

        episode_reward += reward
        episode_length += 1
        if episode_length >= MAX_LEN:
            done = True
    wandb.log({
        "episode reward": episode_reward,
        "solved": int(solved)
    })
