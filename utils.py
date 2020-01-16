import numpy as np
import torch
import gym


import plotly.graph_objects as go


class PointCloud:
    def get_outer(self):
        radius = 0.9285090706636693
        origin = [-0.2678461927495279, -0.006510627535040618,
                  1.0827146125969983]
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2*np.pi, 20)
        x0 = origin[0] + radius * np.outer(np.sin(theta),
                                           np.cos(phi)).flatten()
        y0 = origin[1] + radius * np.outer(np.sin(theta),
                                           np.sin(phi)).flatten()
        z0 = origin[2] + radius * np.outer(np.cos(theta),
                                           np.ones_like(theta)).flatten()
        x = []
        y = []
        z = []

        for xx, yy, zz in zip(x0, y0, z0):
            if zz > 0.85:
                x.append(xx)
                y.append(yy)
                z.append(zz)
        return x, y, z


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class StateBuffer:
    def __init__(self, size):
        self.size = size

    def push(self, partial_state):
        partial_state = torch.tensor(partial_state.copy()).unsqueeze(0)
        self.state = torch.cat((partial_state, self.state[:-1]))
        return self.state

    def reset(self, partial_state):
        partial_state = torch.tensor(partial_state.copy())
        states = [partial_state for _ in range(self.size)]
        self.state = torch.stack(states)
        return self.state


def SkipWrapper(repeat_count):
    class SkipWrapper(gym.Wrapper):
        """
            Generic common frame skipping wrapper
            Will perform action for `x` additional steps
        """

        def __init__(self, env):
            super(SkipWrapper, self).__init__(env)
            self.repeat_count = repeat_count
            self.stepcount = 0

        def step(self, action):
            done = False
            total_reward = 0
            current_step = 0
            while current_step < (self.repeat_count + 1) and not done:
                self.stepcount += 1
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                current_step += 1
            if 'skip.stepcount' in info:
                raise gym.error.Error('Key "skip.stepcount" already in info. Make sure you are not stacking '
                                      'the SkipWrapper wrappers.')
            info['skip.stepcount'] = self.stepcount
            return obs, total_reward, done, info

        def reset(self):
            self.stepcount = 0
            return self.env.reset()

    return SkipWrapper


class LossBuffer:
    def __init__(self, size, discount=0.99):
        self.size = size
        # I added a 0 here so that in the first step the std will not be NaN
        self.returns = [0]
        self.current_return = 0
        self.discount = discount

    def push(self, rew):
        self.current_return = self.discount * self.current_return + rew
        self.push_ret(self.current_return)

    def push_ret(self, ret):
        if len(self.returns) > self.size:
            del self.returns[0]
        self.returns.append(ret)

    def get_std(self):
        returns = torch.tensor(self.returns)
        return returns.std()


class ColorGradient:
    def __init__(self, color2=[220, 36, 36], color1=[74, 86, 157]):
        self.color1 = color1
        self.color2 = color2
        self.colors = zip(color1, color2)

    def get(self, p):
        red = self.color1[0] + p * (self.color2[0] - self.color1[0])
        green = self.color1[1] + p * (self.color2[1] - self.color1[1])
        blue = self.color1[2] + p * (self.color2[2] - self.color1[2])
        return [red, green, blue]


class Plotter3D:
    def __init__(self):
        self.fig = go.Figure()

    def plot_outer_cloud(self, point_cloud):
        x, y, z = point_cloud.get_outer()
        self.fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode='markers',
                marker=dict(
                    size=2,
                    color=1,
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.2
                ))
        )

    def plot_3d_data(self, data):
        x, y, z = zip(*data)
        self.fig.add_trace(
            go.Scatter3d(
                visible=False,
                x=x,
                y=y,
                z=z,
                marker=dict(
                    size=2,
                    color=np.arange(len(x)),
                    colorscale="Viridis",
                    opacity=0.8
                ),
                line=dict(
                    color="lightblue",
                    width=1
                )
            )
        )

    def add_slider(self):
        steps = []
        n_traces = len(self.fig.data)
        for i in range(n_traces):
            step = dict(
                method="restyle",
                args=["visible", [False] * n_traces],
            )
            step["args"][1][i] = True
            step["args"][1][0] = True
            steps.append(step)

        sliders = [dict(
            active=0,
            currentvalue={"prefix": "Frequency: "},
            pad={"t": 50},
            steps=steps
        )]

        self.fig.update_layout(sliders=sliders)

    def show(self):
        self.add_slider()
        self.fig.show()
