from env.environment import Env
from algo.ppo_cont import PPO, Memory
from utils import get_conf, prepare_wandb
from algo.models import ICModule, MultiModalModule
import numpy as np
import wandb

cnf = get_conf("conf/cnf_test.yaml")
env = Env(cnf)
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(action_dim, state_dim)

# prepare logging
wandb.init(config=cnf,
           project="test",
           tags=["MMModule", "test"],
           name="MMModule")

MMModule = MultiModalModule(action_dim, **cnf.MMModel)

state = env.reset()
while True:
    action = agent.policy_old.act(state, memory)
    next_state, *_ = env.step(action)
