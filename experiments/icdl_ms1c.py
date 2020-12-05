"""
Creation of dataset with 10M transitions.
Use both IM and no IM
"""
import torch
import matplotlib.pyplot as plt
from .experiment import BaseExperiment
from evaluate import IVEvaluator
import seaborn as sns
import matplotlib.style as style
from inverse_model import IVModel
import numpy as np
from datetime import datetime

style.use("ggplot")


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)

    def compute_reward(self, state, goal):
        return -(np.linalg.norm(state - goal) ** 2) / 100

    def run(self, pre_run_results):
        device = torch.device("cuda")

        model = IVModel(self.cnf, 2)
        model.load_state_dict(
            torch.load(
                "checkpoints/prod/ckpnt-50000-steps-2-layers-2020-12-02 21:13:06.133274"
            )
        )

        state = self.env.reset()
        for i in range(80):
            state, *_ = self.env.step([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

        goal = state

        state = self.env.reset()

        for i in range(50):
            action = (
                model(
                    torch.tensor(state).float().to(device),
                    torch.tensor(goal).float().to(device),
                )
                .detach()
                .cpu()
                .numpy()
            )
            state, *_ = self.env.step(action)
            print(self.compute_reward(state, goal))
