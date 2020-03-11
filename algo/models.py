"""
Contains models to reproduce the findings from
https://pathak22.github.io/noreward-rl/resources/icml17.pdf
"""
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import LossBuffer

# torch.manual_seed(153)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


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
        # self.fc2 = nn.Linear(128, embedding_size)
        # self.bnorm1 = nn.BatchNorm1d(128)
        # self.bnorm2 = nn.BatchNorm1d(embedding_size)
        self.eval()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bnorm2(F.relu(self.fc2(x)))
        return x.squeeze()


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
        x = torch.cat([x, a])
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
            chain(self._inverse.linear.parameters(),
                  self._inverse.head.parameters(),
                  self._forward.l1.parameters(), self._forward.l2.parameters(),
                  self._forward.head.parameters()))

    def embed(self, state):
        """
        Returns the state embedding from the shared convolutional base.
        """
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
        action = torch.tensor(action).float()
        this_state = torch.tensor(this_state).float()
        next_state = torch.tensor(next_state).float()
        next_state_embed_pred = self.next_state(this_state, action)
        next_state_embed_true = self.embed(next_state)

        self.opt.zero_grad()
        loss = F.mse_loss(next_state_embed_pred, next_state_embed_true)
        loss.backward()
        self.opt.step()
        return loss / 0.02

    def _process_loss(self, loss):
        self.loss_buffer.push(loss)
        return_std = self.loss_buffer.get_std()
        if self.running_return_std is not None:
            self.running_return_std = self.alpha * return_std + (
                1 - self.alpha) * self.running_return_std
        else:
            self.running_return_std = return_std
        return loss / self.running_return_std


class MultiModalModule(nn.Module):
    def __init__(self, action_dim, prop_dim, tac_dim, audio_dim, latent_dim,
                 lstm_hidden_size, lstm_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mods = 3
        self.tac_encoder = ModalityCoder(tac_dim, latent_dim)
        self.prop_encoder = ModalityCoder(prop_dim, latent_dim)
        self.audio_encoder = ModalityCoder(audio_dim, latent_dim)
        self.shared_encoding = nn.Linear(latent_dim + action_dim, latent_dim)
        self.lstm = RecurrentModalModule(latent_dim, lstm_hidden_size,
                                         lstm_layers)
        self.tac_decoder = ModalityCoder(lstm_hidden_size * self.n_mods,
                                         tac_dim)
        self.prop_decoder = ModalityCoder(lstm_hidden_size * self.n_mods,
                                          prop_dim)
        self.audio_decoder = ModalityCoder(lstm_hidden_size * self.n_mods,
                                           audio_dim)

        self.opt = optim.Adam(self.parameters())

    def forward(self, this_state, action):
        action = torch.stack(action).float()
        prop = self.prop_encoder(this_state[0])
        tac = self.tac_encoder(this_state[1])
        audio = self.audio_encoder(this_state[2])

        states = [prop, tac, audio]
        with_action = map(lambda x: torch.cat([x, action], dim=1), states)
        shared_encoding = torch.cat(
            list(map(self.shared_encoding, with_action)))
        lstm_out = self.lstm(
            shared_encoding.view(-1, self.n_mods,
                                 self.latent_dim)).flatten(start_dim=1)
        prop_decoded = self.prop_decoder(lstm_out)
        tac_decoded = self.tac_decoder(lstm_out)
        audio_decoded = self.audio_decoder(lstm_out)
        return prop_decoded, tac_decoded, audio_decoded

    def compute(self, this_state, next_state, action):
        """
        Pass input of the form state: [prop, tac, audio]
        """
        this_state = list(map(torch.cat, this_state))
        next_state = list(map(torch.cat, next_state))
        predicted_states = self.forward(this_state, action)
        loss = 0
        for i, pred_s in enumerate(predicted_states):
            loss += F.mse_loss(pred_s, next_state[i])
        loss.backward()
        self.opt.step()
        return loss


class ModalityCoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class RecurrentModalModule(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size,
                            hidden_size,
                            n_layers,
                            batch_first=True)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out


class MMAE(nn.Module):
    def __init__(self, action_dim, prop_dim, tac_dim, audio_dim, latent_dim,
                 lstm_hidden_size, lstm_layers):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mods = 3
        self.tac_encoder = ModalityCoder(tac_dim + action_dim, latent_dim)
        self.prop_encoder = ModalityCoder(prop_dim + action_dim, latent_dim)
        self.audio_encoder = ModalityCoder(audio_dim + action_dim, latent_dim)
        self.shared_encoding = nn.Linear(latent_dim, latent_dim)
        self.tac_decoder = ModalityCoder(latent_dim, tac_dim)
        self.prop_decoder = ModalityCoder(latent_dim, prop_dim)
        self.audio_decoder = ModalityCoder(latent_dim, audio_dim)

        self.opt = optim.Adam(self.parameters())

    def forward(self, this_state, action):
        action = torch.stack(action).float().to(device)
        this_state = list(
            map(lambda x: torch.cat([x, action], dim=1), this_state))
        prop = self.prop_encoder(this_state[0])
        tac = self.tac_encoder(this_state[1])
        audio = self.audio_encoder(this_state[2])
        states = [prop, tac, audio]
        # with_action = list(map(lambda x: torch.cat([x, action], dim=1),
        #                        states))
        prop_decoded = self.prop_decoder(states[0])
        tac_decoded = self.tac_decoder(states[1])
        audio_decoded = self.audio_decoder(states[2])
        return prop_decoded, tac_decoded, audio_decoded

    def compute(self, trans):
        """
        Pass input of the form state: [prop, tac, audio]
        """
        this_state = [trans.prop, trans.tac, trans.audio]
        next_state = [trans.prop_next, trans.tac_next, trans.audio_next]
        this_state = list(
            map(lambda x: x.to(device), map(torch.cat, this_state)))
        next_state = list(
            map(lambda x: x.to(device), map(torch.cat, next_state)))
        predicted_states = self.forward(this_state, action)
        loss = 0
        for i, pred_s in enumerate(predicted_states):
            loss += F.mse_loss(pred_s, next_state[i])
        loss.backward()
        self.opt.step()
        return loss


if __name__ == "__main__":
    model = MultiModalModule(10, 10, 10, 4, 4, 1, 2)
    bsize = 10
    this_state = [torch.randn(bsize, 10) for i in range(3)]
    next_state = [torch.randn(bsize, 10) for i in range(3)]
    action = torch.randn(bsize, 2)
    loss = model.compute(this_state, next_state, action)
    print(loss)
