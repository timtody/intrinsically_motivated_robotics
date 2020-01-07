import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class A2CAgent:
    def __init__(self, device=None, cnf=None):
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.entropies = []
        self.is_done = False
        self.eps = np.finfo(np.float32).eps.item()
        self.device = device

    def set_policy_network(self, policy_network):
        self.policy = policy_network
        self.opt = torch.optim.Adadelta(self.policy.parameters())

    def append_reward(self, reward):
        self.rewards.append(reward)

    def set_done(self, is_done):
        self.is_done = is_done

    def get_action(self, state):
        # gets the action and stores logprobs and values
        state = torch.tensor(state).float()
        locs, stds, value = self.policy(state)
        stds = torch.eye(stds.size(0)) * stds.squeeze()
        try:
            c = torch.distributions.MultivariateNormal(locs, stds)
        except RuntimeError:
            print(locs, stds)
            raise
        action = c.sample()
        self.entropies.append(c.entropy())
        self.logprobs.append(c.log_prob(action))
        self.values.append(torch.squeeze(value))
        return action

    def train(self):
        self.bootstrap_reward()
        pgloss, vloss = self._compute_losses()
        total_loss = pgloss + vloss
        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()
        self._clear_buffers()
        return pgloss, vloss

    def get_mean_entropy(self):
        return torch.tensor(self.entropies).mean()

    def _compute_losses(self):
        logprobs = torch.stack(self.logprobs)
        values = torch.stack(self.values)
        entropies = torch.stack(self.entropies)
        # compute returns
        #rewards = [sum(self.rewards[i:]) for i in range(len(self.rewards))]
        R = 0
        returns = []
        for reward in self.rewards[::-1]:
            R = 0.99 * R + reward
            returns.insert(0, R)
        returns = torch.tensor(returns).float().to(self.device)

        # normalizing
        returns = self._normalize(returns)
        entropies = self._normalize(entropies)
        values = self._normalize(values)

        advantages = returns - values.detach()
        
        # loss computation
        vloss = F.mse_loss(torch.squeeze(returns), values,
                           reduction="mean") * 0.2
        pgloss = -torch.mean(
            advantages * logprobs) - 0.01 * entropies.mean()
        return pgloss, vloss
    
    def _normalize(self, t):
        """
        :param t: PyTorch Tensor
        """
        t = (t - t.mean()) / (t.std() + self.eps)
        return t

    def bootstrap_reward(self):
        if not self.is_done:
            self.rewards[-1] += self.values[-1]

    def _clear_buffers(self):
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.entropies = []
        self.is_done = False

