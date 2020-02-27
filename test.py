from env.environment import Env
from algo.ppo_cont import PPO, Memory
from utils import get_conf, prepare_wandb
from algo.models import ICModule, MultiModalModule, MMAE
import numpy as np
import torch
from utils import MMBuffer, GraphWindow
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
icmodule = ICModule(cnf.main.action_dim, state_dim, **cnf.icm)
# win = GraphWindow(["reward", "reward raw", "return std", "value_fn"], 1, 4,
#                   10000)
# # tensorboard
writer = SummaryWriter()

timestep = 0
state = env.reset()
i = 0
while True:
    i += 1
    value = 0
    timestep += 1
    action, action_mean = agent.policy_old.act(state.get(), memory)
    next_state, reward, done, _ = env.step(action.numpy())
    # compute im reward
    im_loss = icmodule.train_forward(state.get(), next_state.get(), action)
    im_loss_processed = icmodule._process_loss(im_loss)
    memory.is_terminals.append(done)
    memory.rewards.append(im_loss_processed)
    if timestep % 100 == 0:
        value = agent.policy_old.critic(memory.states[-1])
        agent.update(memory)
        memory.clear_memory()
        timestep = 0

    writer.add_scalar("reward", im_loss_processed, i)
    writer.add_scalar("reward raw", im_loss, i)
    writer.add_scalar("return std", icmodule.loss_buffer.get_std(), i)
    writer.add_scalar("value fn", value, i)
    writer.add_histogram("action mean", action_mean, i)
    # win.update(im_loss_processed, im_loss, icmodule.loss_buffer.get_std(),
    #            value)
