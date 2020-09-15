"""
This file unifies functionality for the PPO agent using the intrinsic curiosity module.
It's supposed to be used to quantitatively analyze the agent's touching behavior
towards it's environment like the table, itself and a pendulum which is in the scene.
"""
import os
import torch
import collections
import numpy as np

from algo.ppo_cont import PPO, Memory
from algo.models import ICModule


class Agent:
    def __init__(self, action_dim, state_dim, cnf, device, is_goal_based=False):
        # PPO related stuff
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.device = device
        self.cnf = cnf
        self.is_goal_based = is_goal_based

        self.init_ppo()
        self.init_icm()

    def init_ppo(self):
        state_dim = self.state_dim if not self.is_goal_based else 2 * self.state_dim
        self.ppo = PPO(self.action_dim, state_dim, self.device, **self.cnf.ppo)
        self.ppo_mem = Memory()

    def init_icm(self):
        self.icm = ICModule(
            self.action_dim, self.state_dim, self.device, **self.cnf.icm
        ).to(self.device)
        self.icm.to(self.device)
        self.icm_buffer = []

    def append_icm_transition(self, this_state, next_state, action) -> None:
        """
        This is for batched training of the ICM. ICM transitions are needed
        for computing the intrinsic reward which is the error when predicting
        s_t+1 from (s_t, a).
        """
        self.icm_buffer.append([this_state, next_state, action])

    def reset_buffers(self) -> None:
        self.icm_buffer = []
        self.ppo_mem.clear_memory()

    def set_alpha(self, val) -> None:
        self.ppo.policy.alpha = val
        self.ppo.policy_old.alpha = val

    def train_ppo(self) -> None:
        self.ppo.update(self.ppo_mem)
        self.ppo_mem.clear_memory()

    def set_is_done(self, is_done) -> None:
        self.ppo_mem.is_terminals.append(is_done)

    def set_reward(self, reward) -> None:
        self.ppo_mem.rewards.append(reward)

    def get_action(self, state, goal=None, inverse_action=None) -> torch.Tensor:
        if goal is not None:
            inverse_action = self.get_inverse_action(state, goal)

        if self.is_goal_based:
            state = np.concatenate([state, goal])

        action, *_ = self.ppo.policy_old.act(state, self.ppo_mem, inverse_action)
        return action

    def get_inverse_action(self, state, goal) -> torch.Tensor:
        return self.icm.get_action(state, goal).squeeze()
    
    def train_forward(self, state, nstate, action, eval=False):
        loss = self.icm.train_forward(state, nstate, action, eval=eval)
        return loss.mean()
        
    def train_with_inverse_reward(self):
        """Trains the inverse model of the agent and uses its loss as the
        intrinsic reward"""
        results = {"ploss": 0, "vloss": 0, "imloss": torch.tensor([0])}

        state_batch, next_state_batch, action_batch = zip(*self.icm_buffer)
        im_loss_batch = self.icm.train_inverse(
            state_batch, next_state_batch, action_batch,
        )
        results["imloss"] = im_loss_batch

        self.ppo_mem.rewards = im_loss_batch

        ploss, vloss = self.ppo.update(self.ppo_mem)
        results["ploss"] = ploss
        results["vloss"] = vloss

        # reset buffers
        self.ppo_mem.clear_memory()
        self.icm_buffer = []

        return results

    def train(
        self, train_fw=True, train_ppo=True, freeze_fw_model=False, random_reward=False
    ) -> dict:
        """
        Trains the ICM of the agent. This method clears the buffer which was filled by
        this.append_icm_transition.
        """
        results = {"ploss": 0, "vloss": 0, "imloss": torch.tensor([0])}

        if train_fw:
            state_batch, next_state_batch, action_batch = zip(*self.icm_buffer)
            im_loss_batch = self.icm.train_forward(
                state_batch, next_state_batch, action_batch, freeze=freeze_fw_model
            )
            results["imloss"] = im_loss_batch

        # train actor
        if train_ppo:
            self.ppo_mem.rewards = (
                im_loss_batch
                if not random_reward
                else im_loss_batch.normal_(mean=28, std=140)
            )
            ploss, vloss = self.ppo.update(self.ppo_mem)
            results["ploss"] = ploss
            results["vloss"] = vloss

        # reset buffers
        self.ppo_mem.clear_memory()
        self.icm_buffer = []

        return results

    def save_state(self, path="") -> None:
        # save icm
        self.icm.save_state(path)
        # save ppo
        self.ppo.save_state(path)

    def load_state(self, path) -> None:
        # load icm
        self.icm.load_state(path)
        # load ppo
        self.ppo.load_state(path)
