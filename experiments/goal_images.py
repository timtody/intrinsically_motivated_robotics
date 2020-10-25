import torch
import numpy as np
import pandas as pd
from .experiment import BaseExperiment
from pyrep.objects.vision_sensor import VisionSensor
import matplotlib.pyplot as plt


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viz = VisionSensor("viz")

    def generate_goals(self, easy=20, medium=20, hard=20):
        easy_range = [9, 10]
        med_range = [24, 25]
        hard_range = [25, 40]
        goals = []
        goals += self._generate_goals(easy, easy_range)
        goals += self._generate_goals(medium, med_range)
        goals += self._generate_goals(hard, hard_range)
        return goals

    def _generate_goals(self, n, goal_range):
        goals = []
        for _ in range(n):
            self.env.reset()
            sign_draw = np.random.choice([-1, 1], size=3)
            horizontal_draw = np.random.randint(goal_range[0], high=goal_range[1])

            for i, sign in enumerate(sign_draw):
                action = [0] * self.cnf.env.action_dim
                action[i] = sign
                for _ in range(horizontal_draw):
                    goal, *_ = self.env.step(action)

            goals.append(goal)
        return goals

    def cap(self, diff):
        im = self.viz.capture_rgb()
        plt.imshow(im)
        plt.axis("off")
        plt.savefig("results/goal_" + diff + ".pdf", bbox_inches="tight")

    def run(self, pre_run):
        self.env.reset()
        for i in range(6):
            self.env.step([0, -1, 0])
        self.cap("start")
        exit(1)
        self.generate_goals(easy=1, medium=0, hard=0)
        self.cap("easy")
        self.generate_goals(easy=0, medium=1, hard=0)
        self.cap("medium")
        self.generate_goals(easy=0, medium=0, hard=1)
        self.cap("hard")

    @staticmethod
    def plot(results, cnf):
        pass
