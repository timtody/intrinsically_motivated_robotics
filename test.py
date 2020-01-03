import torch
from models import ForwardModule, InverseModule



h = w = 42
n_actions = 4

fmodule = ForwardModule(h, w)
imodule = InverseModule(h, w, n_actions)

dummy_data = torch.randn(1, 1, 42, 42)

next_hidden_state = fmodule(dummy_data, torch.tensor([0]).float())
predicted_action = imodule(dummy_data, dummy_data)

print(fmodule.conv_mod.state_dict()["dummy.weight"])
print(imodule.conv_mod.state_dict()["dummy.weight"])