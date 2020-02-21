from env.environment import Env
from algo.ppo_cont import PPO, Memory
from utils import get_conf, prepare_wandb
from algo.models import ICModule, MultiModalModule
import numpy as np
import wandb
import torch

cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(action_dim, state_dim, **cnf.icm)

# prepare logging
wandb.init(config=cnf,
           project="test",
           tags=["MMModule", "test"],
           name="MMModule")

MMModule = MultiModalModule(action_dim, **cnf.MMModel)

state = env.reset()
while True:
    action = agent.policy_old.act(state.get(), memory)
    next_state, *_ = env.step(action)
    action_batch = torch.tensor(action).unsqueeze(0)
    loss = MMModule.compute(state.as_tensor_list(),
                            next_state.as_tensor_list(), action_batch)
    state = next_state
    wandb.log({"loss": loss})
