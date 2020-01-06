import random

import torch
import torch.optim as optim
import torch.nn.functional as F

from matplotlib import pyplot as plt
import seaborn as sns
from models import ICModule

torch.manual_seed(42)
sns.set()

h = w = 42
n_actions = 4

icmodule = ICModule(h, w, n_actions)

STATES = [torch.randn(1, 1, 42, 42) for _ in range(7)]


next_hidden_state = icmodule.next_state(STATES[0], torch.tensor([0]).float())
predicted_action = icmodule.get_action(STATES[0], STATES[0])
optimizer = optim.Adagrad(icmodule.parameters(), lr=1e-4)

def get_action_and_target(states):
    action = random.choice([0, 1, 2, 3, 4, 5, 6])
    target_state = icmodule.embed(states[action])
    action = torch.tensor([action]).float()
    return action, target_state

losses = []
for i in range(50000):
    if i % 5000 == 0:
        STATES = [STATES[-1]] + STATES[:-1]
    action, target = get_action_and_target(STATES)
    next_state = icmodule.next_state(STATES[int(action.item())], action)
    loss = F.mse_loss(next_state, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if i % 100 == 0: print(loss.item())

plt.plot(losses)
plt.show("losses.png")