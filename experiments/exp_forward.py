import matplotlib.pyplot as plt
from .experiment import BaseExperiment
import numpy as np
import pickle
import wandb
import torch
import time
import os


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # experiment parameters
        self.episode_len = 500
        self.dataset = []
        self.ds_path = f"out/forward/ds-rank{self.rank + self.cnf.main.addrank}.p"

    def run(self, pre_run_results):
        self.gen_dataset()

    def train(self):
        train_set, test_set = self.load_dataset()

        for i in range(self.cnf.main.n_steps):
            idx = np.random.randint(len(train_set), size=1000)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.train_forward(state_batch, next_state_batch, action_batch)
            self.wandb.log({"train loss": loss}, step=i)

            if i % 1000 == 0:
                print("evaluating at step", i)
                state_batch, next_state_batch, action_batch = zip(*test_set)
                loss = self.agent.train_forward(
                    state_batch, next_state_batch, action_batch, eval=True
                )
                self.wandb.log({"test loss": loss}, step=i)

    def gen_dataset(self):
        state = self.env.reset()

        for i in range(self.cnf.main.n_steps):
            if i % 10000 == 0:
                print("dataset creation: rank", self.rank, "at step", i)
            action = self.env.action_space.sample()
            nstate, *_ = self.env.step(action)
            self.dataset.append((state, nstate, action))

            state = nstate

        with open(self.ds_path, "wb") as f:
            pickle.dump(self.dataset, f)

    def load_dataset(self):
        with open(self.ds_path.split("-")[0], "rb") as f:
            ds = pickle.load(f)

        split = 0.95
        dataset = np.array(ds)
        np.random.shuffle(dataset)
        test_set = dataset[int(len(dataset) * split) :]
        train_set = dataset[: int(len(dataset) * split)]
        return train_set, test_set
