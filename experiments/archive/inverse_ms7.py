from .experiment import BaseExperiment
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
import pickle
import json
from multiprocessing import Array
import ctypes


class Experiment(BaseExperiment):
    """
    Milestone 7: compute the mean distance per goal to the start state.
    Do this for every difficulty.
    
    Plots: Bar plot showing mean distance to goal.
    Bar plots for easy, medium and hard goal sets.
    Show that hard goals are significantly farther away than easy goals.
    """

    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        # reinit agent for goal based state size
        self.init_agent(is_goal_based=True)
        self.episode_len = 500
        self.results = []
        self.iv_state_template = f"out/inverse_state/ms5/inverse_state_{'im' if self.cnf.main.with_im else 'noim'}.p"

    @staticmethod
    def _load_dataset_im(cnf):
        print("loading dataset im")
        ds = torch.load("out/db-noreset-3dof-im.p")
        # ds = []
        # file_names = glob.glob("out/db/iv_gen_dataset_prop_7dof_with*")
        # for fname in file_names:
        #     with open(fname, "rb") as f:
        #         ds += pickle.load(f)
        # print("done loading")
        return ds

    @staticmethod
    def _load_dataset_noim(cnf):
        print("loading dataset noim")
        ds = torch.load("out/db-noreset-3dof-noim.p")
        # ds = []
        # file_names = glob.glob("out/db/iv_gen_dataset_prop_7dof_no*")
        # for fname in file_names:
        #     with open(fname, "rb") as f:
        #    deviceint("done loading")
        return ds

    def _split_dataset(self, dataset):
        print("splitting dataset")
        split = 0.99
        test_set = dataset[int(len(dataset) * split) :]
        train_set = dataset[: int(len(dataset) * split)]

        print("successfully loaded and splitted dataset of length", len(dataset))
        return train_set, test_set

    def train_inverse_model(self, train_set, test_set):
        print("rank", self.rank, "Training inverse model")
        self.env.reset()

        for i in range(self.cnf.main.iv_train_steps):
            idx = np.random.randint(len(train_set), size=1000)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            action_batch = np.array(action_batch)[:, : self.cnf.env.action_dim]
            loss = self.agent.icm.train_inverse(
                state_batch,
                next_state_batch,
                torch.tensor(action_batch).to(self.device),
                eval=False,
            )

            if i % 1000 == 0:
                print("Rank", self.rank, "evaluating")
                state_batch, next_state_batch, action_batch = zip(*test_set)
                action_batch = np.array(action_batch)[:, : self.cnf.env.action_dim]
                loss = self.agent.icm.train_inverse(
                    state_batch,
                    next_state_batch,
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

    def load_iv_state(self):
        print("Loading inverse state...")
        print(self.iv_state_template)
        self.agent.icm.load_inverse_state(self.iv_state_template)

    def compute_dist(self, state, goal):
        return ((goal - state) ** 2).mean()

    @staticmethod
    def pre_run_hook(*args):
        # do loading of dataset here
        return
        print("pre run hook: loading data sets")
        ds_im = Experiment._load_dataset_im(args[0])
        ds_noim = Experiment._load_dataset_noim(args[0])
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
        dataset_im = np.frombuffer(pre_run_results[0].get_obj(), dtype=np.float32)
        dataset_noim = np.frombuffer(pre_run_results[1].get_obj(), dtype=np.float32)

        print("reshaping data sets")
        # if self.cnf.main.with_im:
        #     print("starting dataset generation")
        #     self._gen_dataset_im()
        # else:
        #     print("starting dataset generation")
        #     self._ge

        dataset_im = dataset_im.reshape(-1, 3, 7)
        dataset_noim = dataset_noim.reshape(-1, 3, 7)

        if self.cnf.main.with_im:
            self.train_inverse_model(*self._split_dataset(dataset_im))
        else:
            self.train_inverse_model(*self._split_dataset(dataset_noim))

    # def generate_goals(self, easy=20, medium=20, hard=20):
    #     easy_range = [0, 10]
    #     med_range = [10, 30]
    #     hard_range = [30, 50]
    #     goals = []
    #     # append easy goals
    #     goals += self._generate_goals(easy, easy_range)
    #     goals += self._generate_goals(medium, med_range)
    #     goals += self._generate_goals(hard, hard_range)
    #     return goals

    # def _generate_goals(self, n, goal_range):
    #     goals = []
    #     for _ in range(n):
    #         self.env.reset()
    #         sign_draw = np.random.choice([-1, 1])
    #         horizontal_draw = np.random.randint(goal_range[0], high=goal_range[1])
    #         for _ in range(horizontal_draw):
    #             self.env.step([sign_draw, 0, 0])
    #         for _ in range(30):
    #             goal, *_ = wandb: Waiting for W&B process to finish, PID 304710

    def generate_goals(self, easy=20, medium=20, hard=20):
        easy_range = [5, 10]
        med_range = [10, 25]
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

    def run(self, pre_run_results):
        results = self.compute_distances()
        return results

    def _compute_distances(self, easy, medium, hard):
        goals = self.generate_goals(easy=easy, medium=medium, hard=hard)
        start = self.env.reset()
        distances = []
        for goal in goals:
            distances.append(np.linalg.norm(goal - start))
        return distances

    def compute_distances(self):
        n = 20
        distances_easy = np.array(self._compute_distances(n, 0, 0))
        distances_medium = self._compute_distances(0, n, 0)
        distances_hard = self._compute_distances(0, 0, n)
        results = pd.DataFrame(
            {"easy": distances_easy, "medium": distances_medium, "hard": distances_hard}
        )
        results = results.melt(
            var_name="Goal difficulty", value_name=r"Mean distance to $s_0$"
        )
        results.to_csv("results/ms7/distance_new.csv")
        exit(1)

    def save_config(self):
        if self.rank == 0:
            results_folder = "/home/julius/projects/curious/results/ms4/"
            with open(results_folder + "config.json", "w") as f:
                json.dump(OmegaConf.to_container(self.cnf, resolve=True), f)

    @staticmethod
    def plot(results, bra):
        pass
