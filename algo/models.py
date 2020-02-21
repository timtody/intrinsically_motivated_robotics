"""
Contains models to reproduce the findings from 
https://pathak22.github.io/noreward-rl/resources/icml17.pdf
"""
from itertools import chain

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from ..utils import LossBuffer

# torch.manual_seed(149)


class ConvModule(nn.Module):
    """
    Provides thes shared convolutional base for the inverse and forward model.
    """
    def __init__(self):
        super(ConvModule, self).__init__()
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
        super(FCModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, embedding_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class ForwardModule(nn.Module):
    """
    Module for learning the forward mapping of state x action -> next state
    """
    def __init__(self, embedding_size, action_dim, base):
        super(ForwardModule, self).__init__()
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
        super(InverseModule, self).__init__()
        self.base = base
        # * 2 because we concatenate two states
        self.linear = nn.Linear(embedding_size * 2, 1024)
        self.head = nn.Linear(1024, action_dim)

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
    def __init__(self, action_dim, state_dim, embedding_size):
        super(ICModule, self).__init__()
        # self._conv_base = ConvModule()
        self.base = FCModule(state_dim, embedding_size)
        # define forward and inverse modules
        self._inverse = InverseModule(embedding_size, action_dim, self.base)
        self._forward = ForwardModule(embedding_size, action_dim, self.base)

        self.opt = optim.RMSprop(self.parameters(),
                                 lr=1e-4,
                                 alpha=0.9,
                                 centered=True)

        self.loss_buffer = LossBuffer(100)

    def forward(self, x):
        raise NotImplementedError

    def parameters(self):
        """
        Returns all parameters of the ICModule, removing unnecessary weights
        from base network.
        """
        return set(
            chain(self._inverse.parameters(), self._forward.parameters()))

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
        return loss

    def _process_loss(self, loss):
        self.loss_buffer.push(loss)
        runinng_std = self.loss_buffer.get_std()
        return loss / (runinng_std + 1e-2)


class MultiModalModule(nn.Module):
    def __init__(self, prop_dim, tac_dim, audio_dim, latent_dim,
                 lstm_hidden_size, lstm_layers, action_dim):
        super().__init__()
        n_states = 3
        self.tac_encoder = ModalityCoder(tac_dim, latent_dim)
        self.prop_encoder = ModalityCoder(prop_dim, latent_dim)
        self.audio_encoder = ModalityCoder(audio_dim, latent_dim)
        self.shared_encoding = nn.Linear(latent_dim + action_dim, latent_dim)
        self.forward_lstm = RecurrentModalModule(latent_dim, lstm_hidden_size,
                                                 lstm_layers)
        self.tac_decoder = ModalityCoder(lstm_hidden_size * n_states, tac_dim)
        self.prop_decoder = ModalityCoder(lstm_hidden_size * n_states,
                                          prop_dim)
        self.audio_decoder = ModalityCoder(lstm_hidden_size * n_states,
                                           audio_dim)

        self.opt = optim.Adam(self.parameters())

    def forward(self, this_state, next_state, action):
        action = torch.tensor(action).float()
        prop = self.prop_encoder(this_state[0])
        tac = self.prop_encoder(this_state[1])
        audio = self.audio_encoder(this_state[2])
        states = [prop, tac, audio]
        with_action = map(lambda x: torch.cat([x, action], dim=1), states)
        stacked = torch.cat(list(map(self.shared_encoding, with_action)))
        lstm_out = self.forward_lstm(stacked.view(-1, 3,
                                                  4)).flatten(start_dim=1)
        prop_decoded = self.prop_decoder(lstm_out)
        tac_decoded = self.prop_decoder(lstm_out)
        audio_decoded = self.prop_decoder(lstm_out)
        return prop_decoded, tac_decoded, audio_decoded

    def compute(self, this_state, next_state, action):
        """
        Pass input of the form state: [prop, tac, audio]
        """
        predicted_states = self.forward(this_state, next_state, action)
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


if __name__ == "__main__":
    model = MultiModalModule(10, 10, 10, 4, 4, 1, 2)
    bsize = 10
    this_state = [torch.randn(bsize, 10) for i in range(3)]
    next_state = [torch.randn(bsize, 10) for i in range(3)]
    action = torch.randn(bsize, 2)
    loss = model.compute(this_state, next_state, action)
    print(loss)
