"""
Use a linear inverse model to learn the inverse problem on 7DOF data sets
"""
import os
import torch
import ctypes
import numpy as np
from .experiment import BaseExperiment
from multiprocessing import Array


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.cnf = cnf
        self.rank = rank
        self.target_folder = "out/ds/off-policy"
        self.iv_state_template = (
            f"out/ds/off-policy/iv-state-rank{self.rank}"
            + f"relu-im-{self.cnf.main.with_im}"
        )

    def train_inverse_model(self, train_set, test_set):
        print("rank", self.rank, "Training inverse model")
        for i in range(self.cnf.main.n_steps):
            idx = np.random.randint(len(train_set), size=self.cnf.main.bsize)
            state_batch, nstate_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.icm.train_inverse(
                state_batch,
                nstate_batch,
                torch.tensor(action_batch).to(self.device),
                eval=False,
            )

            if i % 1000 == 0:
                print("Rank", self.rank, "evaluating")
                state_batch, nstate_batch, action_batch = zip(*test_set)
                loss = self.agent.icm.train_inverse(
                    state_batch,
                    nstate_batch,
                    torch.tensor(action_batch).to(self.device),
                    eval=True,
                )
                self.wandb.log(
                    {f"im: {self.cnf.main.with_im} eval loss": loss.mean()}, step=i
                )

            if i % 50 == 0:
                self.wandb.log(
                    {f"im: {self.cnf.main.with_im} training loss": loss.mean()}, step=i
                )

        self.save_iv_state()

    def save_iv_state(self):
        print("Saving inverse state...")
        self.agent.icm.save_inverse_state(self.iv_state_template)

    def split_dataset(self, dataset):
        print("splitting dataset")
        split = 0.99
        test_set = dataset[int(len(dataset) * split) :]
        train_set = dataset[: int(len(dataset) * split)]

        print("successfully loaded and splitted dataset of length", len(dataset))
        return train_set, test_set

    def create_train_and_test_from_buffer(self, dataset):
        dataset = np.frombuffer(dataset.get_obj(), dtype=np.float32)
        dataset = dataset.reshape(-1, 3, 7)
        train_set, test_set = self.split_dataset(dataset)
        return train_set, test_set

    def run(self, pre_run_results):
        buffer_im, buffer_noim = pre_run_results
        train_im, test_im = self.create_train_and_test_from_buffer(buffer_im)
        train_noim, test_noim = self.create_train_and_test_from_buffer(buffer_noim)
        if self.cnf.main.with_im:
            self.train_inverse_model(train_im, test_im)
        else:
            self.train_inverse_model(train_noim, test_noim)

    @staticmethod
    def pre_run_hook(cnf):
        ds_with_im = Experiment.load_dataset_with_im()
        ds_without_im = Experiment.load_datasset_without_im()

        ds_container_im = Array(ctypes.c_float, ds_with_im.flatten())
        ds_container_noim = Array(ctypes.c_float, ds_without_im.flatten())
        print("Exiting pre run hook")
        return ds_container_im, ds_container_noim

    @staticmethod
    def load_dataset_with_im():
        ds = torch.load("out/ds/off-policy/dataset_with_im")
        return ds

    @staticmethod
    def load_datasset_without_im():
        ds = torch.load("out/ds/off-policy/dataset_without_im")
        return ds
