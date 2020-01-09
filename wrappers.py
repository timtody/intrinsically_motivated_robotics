import gym


class ObsWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        return obs, reward, terminate, None

    def reset(self):
        descriptions, obs = self.task.reset()
        del descriptions  # Not used.
        return obs
