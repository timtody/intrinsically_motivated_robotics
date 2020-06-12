import os
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


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
        # del self.rewards[:]
        self.rewards = []
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(
        self,
        action_dim,
        state_dim,
        n_latent_var,
        max_action,
        action_std,
        device,
        alpha,
    ):
        super(ActorCritic, self).__init__()
        self.alpha = alpha
        self.device = device
        self.action_dim = action_dim
        self.max_action = max_action
        self.actor = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim),
            nn.Tanh(),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.ReLU(),
            nn.Linear(n_latent_var, 1),
        )
        self.action_var = torch.full((action_dim,), action_std * action_std).to(
            self.device
        )

    def action_layer_cont(self, x):
        x = self.action_layer(x)
        locs, stds = self.cont_action(x)
        return locs, stds

    def forward(self):
        raise NotImplementedError

    def get_value(self, state):
        return self.critic(torch.tensor(state).float().to(self.device))

    def act(self, state, memory, inverse_action=None):
        state = torch.from_numpy(state).float().to(self.device)
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()

        if inverse_action is not None:
            action = (1 - self.alpha) * action + self.alpha * torch.tensor(
                inverse_action.detach()
            )

        action_logprob = dist.log_prob(action)
        entropy = dist.entropy()

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(action_logprob)

        # action += torch.randn(self.action_dim) * 0.1

        return (action.detach(), action_mean.detach().cpu().numpy(), entropy.item())

    def evaluate(self, state, action):
        action_mean = self.actor(state)

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(self.device)

        dist = MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy


class PPO:
    def __init__(
        self,
        action_dim,
        state_dim,
        device,
        alpha,
        n_latent_var,
        lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        max_action,
        action_std,
    ):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.device = device

        self.policy = ActorCritic(
            action_dim,
            state_dim,
            n_latent_var,
            max_action,
            action_std,
            self.device,
            alpha,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(
            action_dim,
            state_dim,
            n_latent_var,
            max_action,
            action_std,
            self.device,
            alpha,
        ).to(self.device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def get_value(self, state):
        return self.policy_old.get_value(state).item()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = self.policy.critic(memory.states[-1])

        for reward, is_terminal in zip(
            reversed(memory.rewards), reversed(memory.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(self.device).double()
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.stack(memory.states).to(self.device).detach()
        old_actions = torch.stack(memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(
                old_states, old_actions
            )

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            # for now don't normalize advantages
            # advantages = (advantages - advantages.mean()) / (advantages.std()+
            #                                                  1e-5)

            surr1 = ratios * advantages
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            value_loss = self.MseLoss(state_values.double(), rewards)
            loss = -torch.min(surr1, surr2) + 0.5 * value_loss  # - 1.0 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss.mean(), value_loss

    def load_state(self, path):
        print("ppo loading model from path", path)
        self.policy.load_state_dict(torch.load(os.path.join(path, "policy.pt")))
        self.optimizer.load_state_dict(torch.load(os.path.join(path, "opt.pt")))
        self.policy_old.load_state_dict(torch.load(os.path.join(path, "policy_old.pt")))

    def save_state(self, path):
        print("ppo saving model to path", path)
        torch.save(self.policy.state_dict(), os.path.join(path, "policy.pt"))
        torch.save(self.policy_old.state_dict(), os.path.join(path, "policy_old.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(path, "opt.pt"))
