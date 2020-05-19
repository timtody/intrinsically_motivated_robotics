from .experiment import BaseExperiment
import time
import numpy as np


class Experiment(BaseExperiment):
    def run(self):
        obs = self.env.reset()
        start = time.time()
        for i in range(1000):
            action = np.empty((7,))
            action[:] = np.nan
            obs, *_, info = self.env.step(action)
            print(info)
        end = time.time()
        print("took", end - start, "s")
        self.env.close()
