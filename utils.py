import numpy as np
import torch
import gym


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
    def __init__(self, color1, color2):
        self.color1 = color1
        self.color2 = color2
        self.colors = zip(color1, color2)
    
    def get(self, p):
        red = self.color1[0] + p * (self.color2[0] - self.color2[0])
        green = self.color1[1] + p * (self.color2[1] - self.color2[1])
        blue = self.color1[2] + p * (self.color2[2] - self.color2[2])
        return [red, green, blue]