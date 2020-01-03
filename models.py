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
        return x
    
    def conv2d_size_out(self, size, n_convs, kernel_size=3, stride=2, padding=1):
        for i in range(n_convs):
            size = ((size - (kernel_size - 1) - 1) + padding * 2) // stride  + 1
        return size


shared_conv = ConvModule()


class ForwardModule(nn.Module):
    def __init__(self, h, w):
        super(ForwardModule, self).__init__()
        self.conv_mod = shared_conv
        convw = shared_conv.conv2d_size_out(w, 4)
        convh = shared_conv.conv2d_size_out(h, 4)
        linear_input_size = convw * convh * 32
        # we add + 1 because of the concatenated action
        self.linear = nn.Linear(linear_input_size + 1, 256)
        self.head = nn.Linear(256, linear_input_size)

    def forward(self, x, a):
        x = self.conv_mod(x)
        x = torch.cat([x.flatten(), a])
        x = self.linear(x)
        x = self.head(x)
        return x


class InverseModule(nn.Module):
    def __init__(self, h, w, n_actions):
        super(InverseModule, self).__init__()
        self.conv_mod = shared_conv
        convw = shared_conv.conv2d_size_out(w, 4)
        convh = shared_conv.conv2d_size_out(h, 4)
        linear_input_size = convw * convh * 32
        self.linear = nn.Linear(linear_input_size * 2, 256)
        self.head = nn.Linear(256, n_actions)

    def forward(self, x, y):
        x = self.conv_mod(x)
        y = self.conv_mod(y)
        x = torch.cat([x.flatten(), y.flatten()])
        x = self.linear(x)
        x = self.head(x)
        return x