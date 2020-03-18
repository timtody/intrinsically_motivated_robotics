import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")


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
