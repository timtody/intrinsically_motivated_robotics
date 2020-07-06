from .experiment import BaseExperiment
import altair as alt
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
import pickle
import json
import time

alt.data_transformers.disable_max_rows()


class Experiment(BaseExperiment):
    """
    Implements milestone 5 of the inverse experiments.

    This experiments test the approach with the full 7-DOF
    and analyzes if the approach can actually benefit from intrinsic
    motivations.

    Plot: Show the episode length of an agent with inverse actions
    vs. an agent with random actions vs an agent with inverse actions
    trained from intrinsic motivations.
    """

    grid_size = 17

    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.episode_len = 200
        self.dataset_len = 200000
        self.results = []
        self.iv_state_template = f"out/inverse_state/rank{self.rank}_ms4_2dof_0.pt"

    def _gen_dataset_im(self):
        if not self.rank:
            print("generating data set with length", self.dataset_len)
        # first make the data set
        dataset = []
        state = self.env.reset()

        for i in range(self.dataset_len):
            self.global_step += 1
            if i % 10000 == 0:
                print(f"Data set generation: rank {self.rank} at step", i)
            action = self.agent.get_action(state)

            next_state, _, done, _ = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, action)
            dataset.append((state, next_state, action.tolist()))

            if self.global_step % self.episode_len == self.episode_len - 1:
                done = True
                self.env.reset()

            self.agent.set_is_done(done)
            state = next_state

            if done:
                self.agent.train()

        ds_name = f"out/db/iv_gen_dataset_prop_7dof_with-im_rank{self.rank}.p"
        print("Data set generation: write data (with im) set to", ds_name)
        with open(ds_name, "wb") as f:
            pickle.dump(dataset, f)

    def _gen_dataset_noim(self):
        if not self.rank:
            print("generating data set (no im) with length", self.dataset_len)
        dataset = []
        state = self.env.reset()

        for i in range(self.dataset_len):
            if i % 10000 == 0:
                print(f"Data set generation: rank {self.rank} at step", i)
            action = self.env.action_space.sample()

            next_state, *_ = self.env.step(action)
            dataset.append((state, next_state, list(action)))

            if self.global_step % self.episode_len == self.episode_len - 1:
                self.env.reset()

            state = next_state

        ds_name = f"out/db/iv_gen_dataset_prop_7dof_no-im_rank{self.rank}.p"
        print("Data set generation: write data (no im) set to", ds_name)
        with open(ds_name, "wb") as f:
            pickle.dump(dataset, f)

    def _load_dataset(self):
        with open("out/dataset0_2dof_prop.p", "rb") as f:
            dataset = np.array(pickle.load(f))

        split = 0.95
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        test_set = dataset[int(len(dataset) * split) :]
        train_set = dataset[: int(len(dataset) * split)]

        print("successfully loaded dataset of length", len(dataset))
        return train_set, test_set

    def train_inverse_model(self):
        train_set, test_set = self._load_dataset()
        print("Training inverse model")
        _state = self.env.reset()

        for i in range(self.cnf.main.iv_train_steps):
            idx = np.random.randint(len(train_set), size=500)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.icm.train_inverse(
                state_batch, next_state_batch, torch.tensor(action_batch), eval=False,
            )

            if i % 1000 == 0:
                print("evaluating...")
                state_batch, next_state_batch, action_batch = zip(*test_set)
                loss = self.agent.icm.train_inverse(
                    state_batch,
                    next_state_batch,
                    torch.tensor(action_batch),
                    eval=True,
                )
                self.wandb.log({"eval loss": loss.mean()}, step=i)

            if i % 50 == 0:
                self.wandb.log({"training loss": loss.mean()}, step=i)

        # self.save_iv_state()

    def save_iv_state(self):
        print("Saving inverse state...")
        self.agent.icm.save_inverse_state(self.iv_state_template)

    def load_iv_state(self):
        print("Loading inverse state...")
        self.agent.icm.load_inverse_state(self.iv_state_template)

    def compute_dist(self, state, goal):
        return ((goal - state) ** 2).mean()

    def run(self):
        if self.cnf.main.with_im:
            self._gen_dataset_im()
        else:
            self._gen_dataset_noim()
        return ()

    def save_config(self):
        if self.rank == 0:
            results_folder = "/home/julius/projects/curious/results/ms4/"
            with open(results_folder + "config.json", "w") as f:
                json.dump(OmegaConf.to_container(self.cnf, resolve=True), f)

    @staticmethod
    def plot(results):
        pass
