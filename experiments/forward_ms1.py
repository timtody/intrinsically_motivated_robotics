import torch
import numpy as np
import pandas as pd
from .experiment import BaseExperiment


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        print("bruuhhhh")
        super().__init__(*args, **kwargs)

    def gen_dataset(self):
        state = self.env.reset()
        dataset = []
        train_each = 100
        for i in range(self.cnf.main.n_steps):
            action = self.agent.get_action(state)
            next_state, *_ = self.env.step(action)
            dataset.append((state, next_state, action.numpy()))
            state = next_state
            self.agent.set_is_done(False)

            if i % train_each == train_each - 1:
                reward = self.agent.train(
                    train_fw=False, random_reward=True, length=train_each
                )

        torch.save(
            dataset, f"results/forward/ms0/dataset_{self.cnf.env.state}_rank{self.rank}"
        )

    def run(self, pre_run):
        self.gen_dataset()

    @staticmethod
    def plot(results, cnf):
        pass

