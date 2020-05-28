from .experiment import BaseExperiment
import numpy as np
import torch


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.touch_map = torch.zeros(28)

    def update_touch_map(self):
        self.touch_map += self.env.get_touch_map() / self.env.OBS_SCALER

    def run(self):

        obs = self.env.reset()

        for i in range(1000):
            obs, *_, info = self.env.step(self.env.action_space.sample() * 2)
            print("---")
            self.update_touch_map()
            print(self.touch_map)

        self.env.close()
