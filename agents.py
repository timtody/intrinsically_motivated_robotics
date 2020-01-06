import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class A2CAgent:
    def __init__(self):
        self.policy = None
        self.opt = None

    def get_action(self, state):
        pass

    def train(self):
        pass

    def set_policy_network(self, net):
        self.policy = net
        self.opt = optim.Adadelta(net.parameters())
    