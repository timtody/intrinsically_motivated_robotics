"""
Creation of dataset with 10M transitions.
Use both IM and no IM
"""
import os
import torch
import numpy as np
from .experiment import BaseExperiment


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.cnf = cnf
        self.rank = rank
        self.dataset = []
        self.target_folder = "out/ds/off-policy"
        self.episode_len = 500

    def generate_dataset(self):
        assert (
            self.cnf.env.action_dim == 7
        ), "Action dim needs to be 7 for this experiment"

        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

        if not self.rank:
            print(
                "===Generating dataset===",
                f"IM: {self.cnf.main.with_im}",
                f"Transitions: {self.cnf.main.n_steps}",
                f"Workers: {self.cnf.mp.n_procs}",
                "========================",
                sep="\n",
            )

        if self.cnf.main.with_im:
            self.generate_dataset_with_im()
        else:
            self.generate_dataset_without_im()

    def generate_dataset_without_im(self):
        state = self.env.reset()

        for i in range(self.cnf.main.n_steps):
            if i % 1000 == 0:
                print(f"rank {self.rank} at step", i)
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            self.dataset.append((state, next_state, action))
            state = next_state

            if i % self.episode_len == self.episode_len - 1:
                self.env.reset()
            self.generate_dataset_without_im()reset()

        self.dataset = np.array(self.dataset)

        torch.save(
            self.dataset, self.target_folder + f"/ds_without_im_rank{self.rank}.p"
        )

    def generate_dataset_with_im(self):
        state = self.env.reset()

        for i in range(self.cnf.main.n_steps):
            if i % 1000 == 0:
                print(f"rank {self.rank} at step", i)
            action = self.agent.get_action(state)
            next_state, _, done, _ = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, action)
            self.dataset.append((state, next_state, action.numpy()))
            self.agent.set_is_done(done)
            state = next_state

            if i % self.episode_len == self.episode_len - 1:
                self.env.reset()
                self.agent.train()

        self.dataset = np.array(self.dataset)

        torch.save(self.dataset, self.target_folder + f"/ds_with_im_rank{self.rank}.p")
    

    def run(self, pre_run_results):
        self.generate_dataset()
