import torch
import numpy as np
import pandas as pd
from .experiment import BaseExperiment


class Experiment(BaseExperiment):
    """
    Visualise the performance of the forward model in joint angle space
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, prerun):
        state = self.env.reset()
        col_self = 0
        col_ext = 0

        for i in range(self.cnf.main.n_steps):
            action = self.agent.get_action(state)
            next_state, reward, done, info = self.env.step(action)
            col_self += info["collided_self"]
            col_ext += info["collided_other"]
        return [
            self.rank,
            col_self / self.cnf.main.n_steps,
            col_ext / self.cnf.main.n_steps,
        ]

    @staticmethod
    def plot(results, cnf):
        res = []
        for key, value in results.items():
            res.append(value)
        df = pd.DataFrame(data=res, columns=["Rank", "Self", "Other"],)
        df.to_csv(f"results/forward/ms5/res.csv",)
