import numpy as np
import torch
import gym
import wandb
import plotly.graph_objects as go
from omegaconf import OmegaConf
import numpy as np
import matplotlib.pyplot as plt
from imageio import get_writer
import seaborn as sns
import random
from collections import namedtuple


def get_conf(path):
    # logging and hyperparameters
    cnf = OmegaConf.load(path)
    cnf.merge_with_cli()
    OmegaConf.set_struct(cnf, True)
    return cnf


def prepare_wandb(cnf, *args):
    if cnf.wandb.use:
        wandb.init(project=cnf.wandb.project, name=cnf.wandb.name, config=cnf)
        wandb.watch(args[0].policy, log="all")
        wandb.watch(args[1]._forward, log="all")


class PointCloud:
    def get_outer(self):
        radius = 0.9285090706636693
        origin = [
            -0.2678461927495279, -0.006510627535040618, 1.0827146125969983
        ]
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2 * np.pi, 20)
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


class ReplayBuffer:
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

        return (torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device))


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
                raise gym.error.Error(
                    'Key "skip.stepcount" already in info. Make sure you are not stacking '
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
                    colorscale='Viridis',  # choose a colorscale
                    opacity=0.2)))

    def plot_3d_data(self, data):
        x, y, z = zip(*data)
        self.fig.add_trace(
            go.Scatter3d(visible=True,
                         x=x,
                         y=y,
                         z=z,
                         marker=dict(size=2,
                                     color=np.arange(len(x)),
                                     colorscale="Viridis",
                                     opacity=0.8),
                         line=dict(color="lightblue", width=1)))

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

        sliders = [
            dict(active=0,
                 currentvalue={"prefix": "Frequency: "},
                 pad={"t": 50},
                 steps=steps)
        ]

        self.fig.update_layout(sliders=sliders)

    def show(self):
        # self.add_slider()
        self.fig.show()

    def save(self, fname):
        self.fig.write_html(fname)


class ReturnIAX:
    def __init__(self, ax, discount_factor=0.95, lookback=500):
        self._ax = ax
        self._x = np.arange(-lookback + 1, 1)
        self._return_buffer = np.zeros(lookback)
        self._prediction_buffer = np.zeros(lookback)
        self._target_buffer = np.zeros(lookback)
        self._prediction_part = np.zeros(lookback)
        self._gammas = np.cumprod(np.full(
            lookback, discount_factor))[::-1] / discount_factor
        self._discount_factor = discount_factor
        self._return_curve, = self._ax.plot(self._x,
                                            self._return_buffer,
                                            color="b")
        self._prediction_curve, = self._ax.plot(self._x,
                                                self._prediction_buffer,
                                                color="r")

    def __call__(self, reward, prediction):
        self._prediction_buffer[:-1] = self._prediction_buffer[1:]
        self._prediction_buffer[-1] = prediction
        self._return_buffer[:-1] = self._return_buffer[1:]
        self._return_buffer += self._gammas * reward - self._prediction_part
        self._prediction_part = prediction * self._gammas
        self._return_buffer[-1] = reward
        self._return_buffer += self._discount_factor * self._prediction_part
        self._return_curve.set_ydata(self._return_buffer)
        self._prediction_curve.set_ydata(self._prediction_buffer)


class ForceIAX:
    def __init__(self, ax, lookback=500):
        self._ax = ax
        self._x = np.arange(-lookback + 1, 1)
        self._buffer = np.zeros(lookback)
        self._curve = ax.plot(self._x, self._buffer, color="b")[0]

    def __call__(self, value):
        self._ax.set_ylim(-max(self._buffer) * 1.1, max(self._buffer) * 1.1)
        self._buffer = np.roll(self._buffer, -1)
        self._buffer[-1] = value
        self._curve.set_ydata(self._buffer)


class _ReturnWindow:
    def __init__(self, discount_factor=0.99, lookback=200):
        self.fig, axes = plt.subplots(nrows=5, ncols=3)
        [ax.set_ylim(-5, 5) for ax in axes.flatten()]
        [ax.set_xlim(-lookback, 0) for ax in axes.flatten()]
        self.iaxes = [ForceIAX(ax) for ax in axes.flatten()]
        self._fig_shown = False

    def close(self):
        plt.close(self.fig)

    def update(self, values):
        for value, ax in zip(values, self.iaxes):
            ax(value)
        if not self._fig_shown:
            self.fig.canvas.draw()
            self.fig.show()
            self._fig_shown = True
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def get_frame(self):
        w, h = self.fig.canvas.get_width_height()
        return np.fromstring(self.fig.canvas.tostring_rgb(),
                             dtype=np.uint8).reshape(h, w, 3)


class ReturnWindow:
    def __init__(self):
        self.fig = plt.figure()
        self.iax_return = ReturnIAX(self.fig.add_subplot(211))
        self._fig_shown = False

    def close(self):
        plt.close(self.fig)

    def update(self, reward, prediction):
        self.iax_return(reward, prediction)
        if self._fig_shown:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        else:
            self.fig.show()
            self._fig_shown = True

    # def _get_frame(self):
    #     return np.fromstring(self.fig.canvas.tostring_argb(),
    #                          dtype=np.uint8).reshape(height, width,
    #                                                  4)[:, :,
    #                                                     1:]  # discard alpha


class GraphWindow:
    def __init__(self, labels_list, ncols, nrows, lookback=200):
        sns.set()
        plt.tight_layout()
        plt.autoscale(enable=True, axis='both')
        print("Initializing subplots with", nrows, "rows and", ncols,
              "columns.")
        self.fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        try:
            axes = axes.flatten()
        except Exception:
            raise
        [ax.set_ylim(-5, 5) for ax in axes]
        [ax.set_xlim(-lookback, 0) for ax in axes]
        [ax.set_xlabel("t") for ax in axes]
        [ax.set_title(label) for ax, label in zip(axes, labels_list)]

        self.iaxes = [ForceIAX(ax, lookback=lookback) for ax in axes]
        self._fig_shown = False

    def close(self):
        plt.close(self.fig)

    def update(self, *values):
        for value, ax in zip(values, self.iaxes):
            ax(value)
        if not self._fig_shown:
            self.fig.canvas.draw()
            self.fig.show()
            self._fig_shown = True
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def get_frame(self):
        w, h = self.fig.canvas.get_width_height()
        return np.fromstring(self.fig.canvas.tostring_rgb(),
                             dtype=np.uint8).reshape(h, w, 3)


mm_transition = namedtuple(
    'mm_transition',
    ('prop', 'tac', 'audio', 'prop_next', 'tac_next', 'audio_next', 'action'))


class _MMBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = mm_transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class MMBuffer:
    def __init__(self, tac_dim, prop_dim, audio_dim, action_dim, capacity):
        self.tac_this = np.empty((capacity, tac_dim))
        self.prop_this = np.empty((capacity, prop_dim))
        self.audio_this = np.empty((capacity, audio_dim))
        self.tac_next = np.empty((capacity, tac_dim))
        self.prop_next = np.empty((capacity, prop_dim))
        self.audio_next = np.empty((capacity, audio_dim))
        self.action = np.empty((capacity, action_dim))
        self.pos = 0
        self.cap = capacity
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

    def _normalize(self, entry):
        return (entry - entry.mean()) / (entry.std() + 1e-5)

    def push(self, trans):
        pos = self.pos
        self.tac_this[pos] = trans.tac
        self.prop_this[pos] = trans.prop
        self.audio_this[pos] = trans.audio
        self.tac_next[pos] = trans.tac_next
        self.prop_next[pos] = trans.prop_next
        self.audio_next[pos] = trans.audio_next
        self.action[pos] = trans.action
        self.pos = (pos + 1) % self.cap

    def sample(self, bsize):
        idx = np.random.randint(0, self.cap, size=bsize)
        return mm_transition(
            torch.tensor(self.prop_this[idx]).to(self.device),
            torch.tensor(self.tac_this[idx]).to(self.device),
            torch.tensor(self.audio_this[idx]).to(self.device),
            torch.tensor(self.prop_next[idx]).to(self.device),
            torch.tensor(self.tac_next[idx]).to(self.device),
            torch.tensor(self.audio_next[idx]).to(self.device),
            torch.tensor(self.action[idx]).to(self.device))


if __name__ == "__main__":
    win = ReturnWindow()
    # writer = get_writer("test.mp4", fps=60)

    for i in range(200):
        reward = np.random.randn(15)
        # prediction = np.random.uniform()
        win.update(reward)
