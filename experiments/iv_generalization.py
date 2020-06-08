from .experiment import BaseExperiment
import numpy as np
import torch
import pickle


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = 100

    def run(self):
        # first make the data set
        dataset = []
        state = self.env.reset()

        for i in range(self.n_steps):
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            dataset.append((state, next_state, action))
            state = next_state

        with open(f"out/iv_gen_dataset_rank{self.rank}.pt", "wb") as f:
            pickle.dump(dataset, f)
