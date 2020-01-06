import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvModule(nn.Module):
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
    

class ForwardModule(nn.Module):
    def __init__(self, conv_out_size, conv_base):
        super(ForwardModule, self).__init__()
        self.conv_base = conv_base
        # we add + 1 because of the concatenated action
        self.linear = nn.Linear(conv_out_size + 1, 256)
        self.head = nn.Linear(256, conv_out_size)

    def forward(self, x, a):
        x = self.conv_base(x)
        x = torch.cat([x, a])
        x = self.linear(x)
        x = self.head(x)
        return x


class InverseModule(nn.Module):
    def __init__(self, conv_out_size, conv_base, n_actions):
        super(InverseModule, self).__init__()
        self.conv_base = conv_base
        # * 2 because we concatenate two states
        self.linear = nn.Linear(conv_out_size * 2, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x, y):
        x = self.conv_base(x)
        y = self.conv_base(y)
        x = torch.cat([x, y])
        x = self.linear(x)
        x = self.head(x)
        return x
    

class ICModule:
    def __init__(self, h, w, n_actions):
        self._conv_base = ConvModule()
        convw = shared_conv._conv2d_size_out(w, 4)
        convh = shared_conv._conv2d_size_out(h, 4)
        conv_out_size = convw * convh * 32
        self._inverse = InverseModule(conv_out_size, conv_base, n_actions)
        self._forward = ForwardModule(conv_out_size, conv_base)
    
    def _conv2d_size_out(self, size, n_convs, kernel_size=3, stride=2, padding=1):
        for i in range(n_convs):
            size = ((size - (kernel_size - 1) - 1) + padding * 2) // stride + 1
        return size
    
    def embed(self, state):
        return self._conv_base(state)
    
    def next_state(self, state, action):
        return self._forward(state, action)
    
    def get_action(self, this_state, next_state):
        return self._inverse(this_state, next_state)


class DNNPolicy(nn.Module):
    """
    To be implemented.
    """
    def __init__(self):
        super(DNNPolicy, self).__ini__()
    
    def forward(self, x):
        pass
    

class FCPolicy(nn.Module):
    """
    For testing continuous cart pole before using real env.
    """
    def __init__(self, state_size, n_actions):
        super(FCPolicy, self).__init__()
        self.linear_1 = nn.Linear(state_size, 128)
        self.linear_2 = nn.Linear(128, 128)
        self.logits = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)
    
    def forward(self, x):
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        logits = self.logits(x)
        value = self.value(x)
        probs = F.softmax(logits, dim=-1)
        return probs, value
    