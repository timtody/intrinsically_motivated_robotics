from env.environment import Env
from algo.ppo_cont import PPO, Memory
from utils import get_conf, prepare_wandb
from algo.models import ICModule, MultiModalModule
import numpy as np
import wandb
import torch
from utils import MMBuffer
import pickle

cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
# icmodule = ICModule(action_dim, state_dim, **cnf.icm)

# # prepare logging
# wandb.init(config=cnf,
#            project="test",
#            tags=["MMModule", "test"],
#            name="MMModule")


def save_dataset(ds, name):
    with open(f"data/mm-dataset-{name}.p", "wb") as f:
        pickle.dump(ds, f)


def load_dataset(name):
    with open(f"data/mm-dataset-{name}.p", "rb") as f:
        ds = pickle.load(f)
    return ds


# MMModule = MultiModalModule(action_dim, **cnf.MMModel)
ds_size = 100000
test_set_proportion = 0.2
train_set = MMBuffer(ds_size)
test_set = MMBuffer(int(ds_size * test_set_proportion))
state = env.reset()
for i in range(ds_size):
    action = agent.policy_old.act(state.get(), memory)
    next_state, *_ = env.step(action)
    # action_batch = torch.tensor(action).unsqueeze(0)
    # loss = MMModule.compute(state.as_tensor_list(),
    #                         next_state.as_tensor_list(), action_batch)
    if i % int(1 / test_set_proportion) == 0:
        test_set.push(*state.as_tensor_list(), *next_state.as_tensor_list())
    else:
        train_set.push(*state.as_tensor_list(), *next_state.as_tensor_list())
    state = next_state

save_dataset(train_set, "train")
save_dataset(test_set, "test")
test_set = load_dataset("test")
train_set = load_dataset("train")
print("test set length", len(test_set.memory))
print("train set length", len(train_set.memory))
