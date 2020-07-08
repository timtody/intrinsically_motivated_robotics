from .experiment import BaseExperiment
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
import pickle
import json
import glob
from multiprocessing import Array
import ctypes


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
        self.dataset_len = 1000000
        self.results = []
        self.iv_state_template = f"out/inverse_state/ms5/rank{self.rank}_7dof{self.cnf.main.iv_train_steps}steps.p"

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

    @staticmethod
    def _load_dataset_im():
        print('loading dataset im')
        ds = []
        file_names = glob.glob('out/db/iv_gen_dataset_prop_7dof_with*')
        for fname in file_names:
            with open(fname, 'rb') as f:
                ds += pickle.load(f)
        print('done loading')
        return ds

    @staticmethod
    def _load_dataset_noim():
        print('loading dataset noim')
        ds = []
        file_names = glob.glob('out/db/iv_gen_dataset_prop_7dof_no*')
        for fname in file_names:
            with open(fname, 'rb') as f:
                ds += pickle.load(f)
        print('done loading')
        return ds

    def _split_dataset(self, dataset):
        print("splitting dataset")
        split = 0.99
        test_set = dataset[int(len(dataset) * split):]
        train_set = dataset[: int(len(dataset) * split)]

        print("successfully loaded and splitted dataset of length", len(dataset))
        return train_set, test_set

    def train_inverse_model(self, train_set, test_set):
        print("Training inverse model")
        self.env.reset()

        for i in range(self.cnf.main.iv_train_steps):
            idx = np.random.randint(len(train_set), size=1000)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.icm.train_inverse(
                torch.tensor(state_batch),
                torch.tensor(next_state_batch),
                torch.tensor(action_batch), eval=False,
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
                self.wandb.log(
                    {f"im: {self.cnf.main.with_im} eval loss": loss.mean()}, step=i)

            if i % 50 == 0:
                self.wandb.log(
                    {f"im: {self.cnf.main.with_im} training loss": loss.mean()}, step=i)

        self.save_iv_state()

    def save_iv_state(self):
        print("Saving inverse state...")
        self.agent.icm.save_inverse_state(self.iv_state_template)

    def load_iv_state(self):
        print("Loading inverse state...")
        self.agent.icm.load_inverse_state(self.iv_state_template)

    def compute_dist(self, state, goal):
        return ((goal - state) ** 2).mean()

    @staticmethod
    def pre_run_hook():
        # do loading of dataset here
        print("pre run hook: loading data sets")
        ds_im = Experiment._load_dataset_im()
        ds_noim = Experiment._load_dataset_noim()
        print("pre run hook: converting to numpy")
        ds_im = np.array(ds_im)
        ds_noim = np.array(ds_noim)
        print("pre run hook: creating shared memory arrays")
        ds_container_im = Array(ctypes.c_float, ds_im.flatten())
        ds_container_noim = Array(ctypes.c_float, ds_noim.flatten())
        print("Exiting pre run hook")
        return ds_container_im, ds_container_noim

    def train_models(self, pre_run_results):
        print("making objects from buffers")
        dataset_im = np.frombuffer(
            pre_run_results[0].get_obj(), dtype=np.float32)
        dataset_noim = np.frombuffer(
            pre_run_results[1].get_obj(), dtype=np.float32)

        print("reshaping data sets")

        dataset_im = dataset_im.reshape(-1, 3, 7)
        dataset_noim = dataset_noim.reshape(-1, 3, 7)

        if self.cnf.main.with_im:
            self.train_inverse_model(*self._split_dataset(dataset_im))
        else:
            self.train_inverse_model(*self._split_dataset(dataset_noim))

    def run(self, pre_run_results):
        if self.rank % 2 == 0:
            self.cnf.main.with_im = True
        else:
            self.cnf.main.with_im = False

        self.train_models(pre_run_results)
        # self.load_iv_state()
        # self.test_performance()
        return ()

    def test_performance(self):
        # acquire goal first
        for i in range(30):
            self.env.step([1, 0, 0, 0, 0, 0, 0])
        for i in range(40):
            goal, *_ = self.env.step([0, 1, 0, 0, 0, 0, 0])

        for i in range(self.cnf.main.n_steps):
            state = self.env.reset()

            ep_len = 0
            reward_sum = 0
            done = False
            while not done:
                reward = 0

                action = self.agent.get_action(state, goal)
                state, *_ = self.env.step(action)
                dist = self.compute_dist(state, goal)
                print(dist)

                ep_len += 2
                if dist < 1:
                    print("we did it!!!!!!!!!!11")
                    done = True
                    reward = 10

                if ep_len > 500:
                    done = True
                    reward = 0

                reward_sum += reward
                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

            self.agent.train_ppo()
            if i % 10 == 0:
                self.results.append(
                    (self.rank, i, ep_len, self.cnf.main.with_im,))

            self.wandb.log(
                {f"im: {self.cnf.main.with_im} episode length": ep_len})

        self.save_config()

        return self.results

    def save_config(self):
        if self.rank == 0:
            results_folder = "/home/julius/projects/curious/results/ms4/"
            with open(results_folder + "config.json", "w") as f:
                json.dump(OmegaConf.to_container(self.cnf, resolve=True), f)

    @staticmethod
    def plot(results):
        pass
