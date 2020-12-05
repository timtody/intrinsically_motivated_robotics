import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FWModel(nn.Module):
    def __init__(self, cnf, delta=False):
        super().__init__()
        self.delta = delta
        self.device = torch.device("cuda" if cnf.main.gpu else "cpu")
        self.hidden = 256
        self.linear = nn.Linear(14, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.head = nn.Linear(self.hidden, 7)
        self.opt = optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)

    def forward(self, x, y):
        """predicts state x action -> next state / delta next state
        x: state
        y: action
        """
        if x.dim() > 1:
            x = torch.cat([x, y], dim=1)
        else:
            x = torch.cat([x, y], dim=0)
        x = F.relu(self.linear(x))
        x = F.relu(self.linear2(x))
        return self.head(x)

    def train(self, states, nstates, actions, eval=False):
        actions = torch.tensor(actions).float().to(self.device)
        this_state = torch.tensor(states).float().to(self.device)
        next_state = torch.tensor(nstates).float().to(self.device)
        if self.delta:
            nstates = nstates - states

        self.opt.zero_grad()
        predicted_nstates = self.forward(states, actions)
        loss = F.mse_loss(predicted_nstates, nstates)
        if not eval:
            loss.backward
            self.opt.step()
        return loss


class IVModel(nn.Module):
    def __init__(self, cnf, depth, act=F.relu, delta=False):
        super().__init__()
        self.delta = delta
        self.device = torch.device("cuda" if cnf.main.gpu else "cpu")
        self.hidden = 256
        self.act = act
        self.depth = depth
        self.linear = nn.Linear(cnf.icm.embedding_size * 2, self.hidden)
        self.linear2 = nn.Linear(self.hidden, self.hidden)
        self.linear3 = nn.Linear(self.hidden, self.hidden)
        self.linear4 = nn.Linear(self.hidden, self.hidden)
        self.head = nn.Linear(self.hidden, cnf.env.action_dim)
        self.opt = optim.Adam(self.parameters(), lr=0.001)
        self.to(self.device)

    def forward(self, x, y):
        if self.delta:
            y = y - x
        if x.dim() > 1:
            x = torch.cat([x, y], dim=1)
        else:
            x = torch.cat([x, y], dim=0)

        x = self.act(self.linear(x))

        if self.depth > 1:
            x = self.act(self.linear2(x))
        if self.depth > 2:
            x = self.act(self.linear3(x))
        if self.depth > 3:
            x = self.act(self.linear3(x))
        return self.head(x)

    def train(self, states, nstates, actions, eval=False):
        if self.delta:
            nstates = nstates - states
        actions = torch.tensor(actions).float().to(self.device)
        this_state = torch.tensor(states).float().to(self.device)
        next_state = torch.tensor(nstates).float().to(self.device)

        self.opt.zero_grad()
        predicted_action = self.forward(this_state, next_state)
        loss = F.mse_loss(predicted_action, actions)
        if not eval:
            loss.backward()
            self.opt.step()
        return loss
