import torch
import torch.optim as optim
import torch.nn.functional as F

import gym
import random
from matplotlib import pyplot as plt
from models import ForwardModule, InverseModule, shared_conv

torch.manual_seed(42)

h = w = 42
n_actions = 4

fmodule = ForwardModule(h, w)
imodule = InverseModule(h, w, n_actions)

dummy_data = torch.randn(1, 1, 42, 42)
dummy_data_2 = torch.randn(1, 1, 42, 42)
dummy_data_3 = torch.randn(1, 1, 42, 42)


next_hidden_state = fmodule(dummy_data, torch.tensor([0]).float())
predicted_action = imodule(dummy_data, dummy_data)

optimizer = optim.Adagrad(fmodule.parameters(), lr=1e-4)

print(shared_conv.state_dict()["conv1.bias"])
print(fmodule.conv_base.state_dict()["conv1.bias"])

states = [dummy_data, dummy_data_2, dummy_data_3]

def get_action_and_target(states):
    action = random.choice([0, 1, 2])
    target_state = shared_conv(states[action])
    action = torch.tensor([action]).float()
    return action, target_state

losses = []
for i in range(3000):
    if i == 1000:
        states = [states[-1]] + states[:-1]
    if i == 2000:
        states = [states[-1]] + states[:-1] 
    action, target = get_action_and_target(states)
    next_state = fmodule(states[int(action.item())], action)
    loss = F.mse_loss(next_state, target)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if i % 100 == 0: print(loss.item())

plt.plot(losses)
plt.savefig("losses.png")
print(shared_conv.state_dict()["conv1.bias"])
print(fmodule.conv_base.state_dict()["conv1.bias"])
