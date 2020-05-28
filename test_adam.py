import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        return x


net = Net()
opt = optim.Adam(params=net.parameters())

out = net(torch.randn(64))
loss = ((out - torch.randn(64)) ** 2).mean()
opt.zero_grad()
loss.backward()
opt.step()

# out = net(torch.randn(64))
# loss = ((out - torch.randn(64)) ** 2).mean()
# opt.zero_grad()
# loss.backward()
# opt.step()

# out = net(torch.randn(64))
# loss = ((out - torch.randn(64)) ** 2).mean()
# opt.zero_grad()
# loss.backward()
# opt.step()

for key, value in opt.state.items():
    print(value["exp_avg"])
    print(value["exp_avg_sq"])
#     if not np.isnan(state.detach()).any():
#         print("no")
