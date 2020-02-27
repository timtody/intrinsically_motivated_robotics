from env.environment import Env
from algo.ppo_cont import PPO, Memory
from utils import get_conf, prepare_wandb
from algo.models import ICModule, MultiModalModule, MMAE
import numpy as np
import torch
from utils import MMBuffer
import pickle
from collections import namedtuple
from torch.utils.tensorboard import SummaryWriter

cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")

# tensorboard
writer = SummaryWriter()

trans = namedtuple(
    'trans',
    ('prop', 'tac', 'audio', 'prop_next', 'tac_next', 'audio_next', 'action'))


def save_dataset(ds, name):
    with open(f"data/mm-dataset-{name}.p", "wb") as f:
        pickle.dump(ds, f)


def load_dataset(name):
    with open(f"data/mm-dataset-{name}.p", "rb") as f:
        ds = pickle.load(f)
    return ds


# MMModule = MMAE(action_dim, **cnf.MMModel).to(device)
# n_epochs = 5000
# bsize = 10000
# timestep = 0
set_size = 100000
test_set_proportion = 0.2
train_set = MMBuffer(cnf.MMModel.tac_dim, cnf.MMModel.prop_dim,
                     cnf.MMModel.audio_dim, action_dim,
                     int((1 - test_set_proportion) * set_size))
test_set = MMBuffer(cnf.MMModel.tac_dim, cnf.MMModel.prop_dim,
                    cnf.MMModel.audio_dim, action_dim,
                    int(test_set_proportion * set_size))
state = env.reset()
for i in range(set_size):
    if i % 50000 == 0:
        print("Currently at iteration:", i)
    action, action_mean = agent.policy_old.act(state.get(), memory)
    next_state, *_ = env.step(action.numpy())
    transition = trans(state.get_prop(), state.get_tac(), state.get_audio(),
                       next_state.get_prop(), next_state.get_tac(),
                       next_state.get_audio(), action)
    if i % int(1 / test_set_proportion) == 0:
        test_set.push(transition)
    else:
        train_set.push(transition)
    state = next_state

print("Train set len:", train_set.cap)
print("Test set len:", test_set.cap)
# print(train_set.sample(1))
# for i in range(n_epochs):
#     for j in range(len(train_set.memory) // bsize):
#         timestep += 1
#         batch = trans(*zip(*train_set.sample(bsize)))
#         loss = MMModule.compute(
#             [batch.prop, batch.tac, batch.audio],
#             [batch.prop_next, batch.tac_next, batch.audio_next], batch.action)
#         writer.add_scalar('Loss/train', loss, timestep)

# state = env.reset()
# for i in range(ds_size):
#     if i % 10000 == 0:
#         print("Iteration", i)
#     action = agent.policy_old.act(state.get(), memory)
#     next_state, *_ = env.step(action)
#     # action_batch = torch.tensor(action).unsqueeze(0)
#     # loss = MMModule.compute(state.as_tensor_list(),
#     #                         next_state.as_tensor_list(), action_batch)
#     if i % int(1 / test_set_proportion) == 0:
#         test_set.push(*state.as_tensor_list(), *next_state.as_tensor_list(),
#                       action)
#     else:
#         train_set.push(*state.as_tensor_list(), *next_state.as_tensor_list(),
#                        action)
#     state = next_state
print("saving datasets")
save_dataset(train_set, "train")
save_dataset(test_set, "test")
