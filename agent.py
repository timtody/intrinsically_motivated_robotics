import os
import torch
import collections

from algo.ppo_cont import PPO, Memory
from algo.models import ICModule


class Agent:
    def __init__(self, action_dim, state_dim, cnf, device):
        # PPO related stuff
        print(action_dim)
        self.ppo = PPO(action_dim, state_dim, **cnf.ppo)
        self.ppo_mem = Memory()

        # ICM related stuff
        self.icm = ICModule(action_dim, state_dim, **cnf.icm).to(device)
        self.icm_transition = collections.namedtuple(
            "icm_trans", ["state", "next_state", "action"]
        )
        self.icm_buffer = []

    def append_icm_transition(self, this_state, next_state, action):
        self.icm_buffer.append(self.icm_transition(this_state, next_state, action))

    def set_is_done(self, is_done):
        self.ppo_mem.is_terminals.append(is_done)

    def set_reward(self, reward):
        self.ppo_mem.rewards.append(reward)

    def get_action(self, state) -> torch.Tensor:
        action, *_ = self.ppo.policy_old.act(state, self.ppo_mem)
        return action

    def train(self) -> None:
        # TODO: return results dict
        # train icm
        state_batch, next_state_batch, action_batch = zip(*self.icm_buffer)
        im_loss_batch = self.icm.train_forward(
            state_batch, next_state_batch, action_batch
        )
        # train actor
        self.ppo_mem.rewards = im_loss_batch.cpu().numpy()
        self.ppo.update(self.ppo_mem)
        self.ppo_mem.clear_memory()
        self.icm_buffer = []

    def save_state(self, timestep) -> None:
        # save icm
        self.icm.save_state(timestep)
        # save ppo
        self.ppo.save_state(timestep)

    def load_state(self, path) -> None:
        # load icm
        self.icm.load_state(path)
        # load ppo
        self.ppo.load_state(path)
