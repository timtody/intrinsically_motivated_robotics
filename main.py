import gym
import rlbench.gym
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from models import ICModule
from ppo import PPO, Memory

cnf = OmegaConf.load("conf/conf.yaml")
cnf.merge_with_cli()

env = gym.make(cnf.main.env_name)

# move to cnf file
obs_space = env.observation_space.shape[0]
n_actions = 2187

agent = PPO(obs_space, n_actions, **cnf.ppo)
memory = Memory()
icmodule = ICModule(obs_space, 1, n_actions)

if cnf.main.use_wb:
    wandb.init(project=cnf.wandb.project, name=cnf.wandb.name)
    wandb.watch(policy, log="all")

state = env.reset()
done = False
for i in range(cnf.main.max_timesteps):
    action = agent.policy_old.act(state, memory)
    print(action)
    exit(1)
    state, _, done, _ = env.step(action)
    loss = icmodule.train_forward(state, next_state, action)
    # IM loss = reward currently
    reward = loss
    agent.append_reward(reward)
    