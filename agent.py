"""
This file unifies functionality for the PPO agent using the intrinsic curiosity module.
It's supposed to be used to quantitatively analyze the agent's touching behavior
towards it's environment like the table, itself and a pendulum which is in the scene.
"""
import os
import torch
import collections

from algo.ppo_cont import PPO, Memory
from algo.models import ICModule


class Agent:
    def __init__(self, action_dim, state_dim, cnf, device):
        # PPO related stuff
        self.ppo = PPO(action_dim, state_dim, **cnf.ppo)
        self.ppo_mem = Memory()

        # ICM related stuff
        self.icm = ICModule(action_dim, state_dim, **cnf.icm).to(device)
        self.icm_transition = collections.namedtuple(
            "icm_trans", ["state", "next_state", "action"]
        )
        self.icm_buffer = []

    def append_icm_transition(self, this_state, next_state, action) -> None:
        """
        This is for batched training of the ICM. ICM transitions are needed
        for computing the intrinsic reward which is the error when predicting
        s_t+1 from (s_t, a).
        """
        self.icm_buffer.append(self.icm_transition(this_state, next_state, action))

    def set_is_done(self, is_done) -> None:
        self.ppo_mem.is_terminals.append(is_done)

    def set_reward(self, reward) -> None:
        self.ppo_mem.rewards.append(reward)

    def get_action(self, state) -> torch.Tensor:
        action, *_ = self.ppo.policy_old.act(state, self.ppo_mem)
        return action

    def train(self) -> dict:
        """
        Trains the ICM of the agent. This method clears the buffer which was filled by
        this.append_icm_transition
        """
        state_batch, next_state_batch, action_batch = zip(*self.icm_buffer)
        im_loss_batch = self.icm.train_forward(
            state_batch, next_state_batch, action_batch
        )
        # train actor
        self.ppo_mem.rewards = im_loss_batch.cpu().numpy()
        ploss, vloss = self.ppo.update(self.ppo_mem)
        self.ppo_mem.clear_memory()
        self.icm_buffer = []

        return {"ploss": ploss, "vloss": vloss, "imloss": im_loss_batch}

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
