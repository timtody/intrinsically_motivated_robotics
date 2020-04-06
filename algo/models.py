"""
Contains models to reproduce the findings from
https://pathak22.github.io/noreward-rl/resources/icml17.pdf
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from itertools import chain

from utils import LossBuffer

torch.manual_seed(1)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class ConvModule(nn.Module):
    """
    Provides thes shared convolutional base for the inverse and forward model.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        return x.flatten()

    @staticmethod
    def _conv2d_size_out(size, n_convs, kernel_size=3, stride=2, padding=1):
        for _ in range(n_convs):
            size = ((size - (kernel_size - 1) - 1) + padding * 2) // stride + 1
        return size


class FCModule(nn.Module):
    """
    Docstring: todo
    """

    def __init__(self, state_dim, embedding_size):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, embedding_size)
        self.fc2 = nn.Linear(128, embedding_size)
        # self.bnorm1 = nn.BatchNorm1d(128)
        # self.bnorm2 = nn.BatchNorm1d(embedding_size)
        self.eval()

    def forward(self, x):
        x = self.fc1(x)
        return x.squeeze(0)


class ForwardModule(nn.Module):
    """
    Module for learning the forward mapping of state x action -> next state
    """

    def __init__(self, embedding_size, action_dim, base):
        super().__init__()
        self.base = base
        # we add + 1 because of the concatenated action
        self.l1 = nn.Linear(embedding_size + action_dim, 128)
        self.l2 = nn.Linear(128, 128)
        self.head = nn.Linear(128, embedding_size)

    def forward(self, x, a):
        with torch.no_grad():
            x = self.base(x)
        x = torch.cat([x, a], dim=1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.head(x)
        return x


class InverseModule(nn.Module):
    """
    Module for learning the inverse mapping of state x next state -> action.
    """

    def __init__(self, embedding_size, action_dim, base):
        super().__init__()
        self.base = base
        # * 2 because we concatenate two states
        self.linear = nn.Linear(embedding_size * 2, 256)
        self.head = nn.Linear(256, action_dim)

    def forward(self, x, y):
        x = self.base(x)
        y = self.base(y)
        x = torch.cat([x, y])
        x = self.linear(x)
        x = self.head(x)
        return x


class ICModule(nn.Module):
    """
    Intrinsic curiosity module.
    """

    def __init__(self, action_dim, state_dim, embedding_size, alpha=0.001):
        super().__init__()
        # self._conv_base = ConvModule()
        self.base = FCModule(state_dim, embedding_size)
        # define forward and inverse modules
        self._inverse = InverseModule(embedding_size, action_dim, self.base)
        self._forward = ForwardModule(embedding_size, action_dim, self.base)

        self.opt = optim.Adam(self.parameters(), lr=5e-5)
        self.loss_buffer = LossBuffer(100)
        self.running_return_std = None
        self.alpha = alpha

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        """
        Returns all parameters of the ICModule, removing unnecessary weights
        from base network.
        """
        return set(
            chain(
                self._inverse.linear.parameters(),
                self._inverse.head.parameters(),
                self._forward.l1.parameters(),
                self._forward.l2.parameters(),
                self._forward.head.parameters(),
            )
        )

    def embed(self, state):
        """
        Returns the state embedding from the shared convolutional base.
        """
        return self.base(state)

    def get_embedding(self, state):
        """
        Returns the state embedding from the shared convolutional base.
        From numpy.
        """
        state = torch.tensor(state).float().to(device)
        return self.base(state)

    def next_state(self, state, action):
        """
        Given state and action, predicts the next state in embedding space.
        """
        return self._forward(state, action)

    def get_action(self, this_state, next_state):
        """
        Given two states, predicts the action taken.
        """
        return self._inverse(this_state, next_state)

    def train_forward(self, this_state, next_state, action):
        action = torch.stack(action).float().to(device)
        this_state = torch.tensor(this_state).float().to(device)
        next_state = torch.tensor(next_state).float().to(device)
        next_state_embed_pred = self.next_state(this_state, action)
        next_state_embed_true = self.embed(next_state)

        self.opt.zero_grad()
        loss = F.mse_loss(
            next_state_embed_pred, next_state_embed_true, reduction="none"
        )
        loss.mean().backward()
        self.opt.step()
        return loss.mean(dim=1).detach() / 0.4065

    def _process_loss(self, loss):
        self.loss_buffer.push(loss)
        return_std = self.loss_buffer.get_std()
        if self.running_return_std is not None:
            self.running_return_std = (
                self.alpha * return_std + (1 - self.alpha) * self.running_return_std
            )
        else:
            self.running_return_std = return_std
        return loss / self.running_return_std

    def save_state(self, timestep):
        torch.save(
            self.state_dict(), os.path.join("checkpoints", str(timestep), "icm.pt")
        )

    def load_state(self, path):
        abspath = os.path.abspath(os.environ["owd"])
        path = os.path.join(abspath, path, "icm.pt")
        print("loading model state from", path)
        self.load_state_dict(torch.load(path))
