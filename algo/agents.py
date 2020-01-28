import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class CURAgent:
    def __init__(self):
        self.policy = None
        self.opt = None
        self.rewards = []
        self.entropies = []
        self.logprobs = []
        self.values = []

    def get_action(self, state):
        state = torch.tensor(state)
        probs, value = self.policy(state)
        c = torch.distributions.Categorical(probs)
        action = c.sample()
        self.entropies.append(c.entropy())
        self.logprobs.append(c.log_prob(action))
        self.values.append(value.squeeze())

    def train(self):
        pass

    def _compute_loss(self):
        pass

    def set_policy_network(self, net):
        self.policy = net
        self.opt = optim.Adadelta(net.parameters())

    def _clear_buffers(self):
        self.rewards = []
        self.logprobs = []
        self.values = []
        self.entropies = []
        self.is_done = False
    