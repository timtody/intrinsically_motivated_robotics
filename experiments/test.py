from .experiment import BaseExperiment
import time


class Experiment(BaseExperiment):
    def run(self):
        obs = self.env.reset()
        start = time.time()
        for i in range(10000):
            self.env.step(self.env.action_space.sample())
        end = time.time()
        print("took", end - start, "s")
        self.env.close()
