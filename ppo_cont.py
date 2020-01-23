import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
torch.manual_seed(42)


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ContActionLayer(nn.Module):
    def __init__(self, n_latent_var, action_dim):
        super(ContActionLayer, self).__init__()
        self.locs = nn.Linear(n_latent_var, action_dim)
        self.stds = nn.Linear(n_latent_var, action_dim)

    def forward(self, x):
        locs = self.locs(x)
        stds = F.softplus(self.stds(x))
        return locs, stds


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, max_action):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.max_action = max_action
        self.cont_action = ContActionLayer(n_latent_var, action_dim)
        # actor
        self.action_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh()
        )

        # critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def action_layer_cont(self, x):
        x = self.action_layer(x)
        locs, stds = self.cont_action(x)
        return locs, stds

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        locs, stds = self.action_layer_cont(state)
        stds = torch.eye(self.action_dim) * stds
        dist = MultivariateNormal(locs, stds)
        action = dist.sample()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(dist.log_prob(action))
        return action.clamp(-self.max_action, self.max_action).numpy()
        # return torch.tanh(action).numpy() * self.max_action

    def evaluate(self, state, action):
        locs, stds = self.action_layer_cont(state)
        # construct batch of diagonal matrices from stds
        covariance_matrix = torch.diag_embed(stds)
        dist = MultivariateNormal(locs, covariance_matrix)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(self,
                 state_dim,
                 action_dim,
                 n_latent_var,
                 lr,
                 betas,
                 gamma,
                 K_epochs,
                 eps_clip,
                 max_action):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(
            state_dim, action_dim, n_latent_var, max_action).to(device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=lr, betas=betas)
        self.policy_old = ActorCritic(
            state_dim, action_dim, n_latent_var, max_action).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip,
                                1+self.eps_clip) * advantages

            loss = -torch.min(surr1, surr2) + 0.5 * \
                self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
