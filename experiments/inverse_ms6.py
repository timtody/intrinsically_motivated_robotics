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
    Milestone 6: explore if IM can improve inverse model learning.
    
    Plots: Bar plot showing mean episode length for alpha = 1
    Bar plots for easy, medium and hard goal sets.
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

    def generate_goals(self, easy=20, medium=20, hard=20):
        easy_range = [0, 7]
        med_range = [8, 20]
        hard_range = [20, 30]
        goals = []
        # append easy goals
        goals += self._generate_goals(easy, easy_range)
        goals += self._generate_goals(medium, med_range)
        goals += self._generate_goals(hard, hard_range)
        return goals

    def _generate_goals(self, n, goal_range):
        goals = []
        for _ in range(n):
            self.env.reset()
            sign_draw = np.random.choice([-1, 1])
            horizontal_draw = np.random.randint(goal_range[0], high=goal_range[1])
            for _ in range(horizontal_draw):
                self.env.step([sign_draw, 0, 0])
            for _ in range(30):
                goal, *_ = self.env.step([0, 1, 0])
            goals.append(goal)
        return goals

    def run(self, pre_run_results):
        self.load_iv_state()
        results = self.test_performance()
        return results

    def test_performance(self):
        goals = self.generate_goals(easy=20, medium=0, hard=0)

        total_episodes = 0

        for i in range(self.cnf.main.n_steps):
            state = self.env.reset()
            # choose goal
            goal_idx = np.random.randint(len(goals))
            goal = goals[goal_idx]

            ep_len = 0
            reward_sum = 0
            done = False
            while not done:
                total_episodes += 1
                reward = 0

                action = self.agent.get_action(state, goal)
                state, *_ = self.env.step(action)
                dist = self.compute_dist(state, goal)
                # print(dist)

                ep_len += 1
                if dist < 1:
                    done = True
                    reward = 1

                if ep_len > 1000:
                    done = True
                    reward = 0

                reward_sum += reward
                # self.agent.set_reward(reward)
                # self.agent.set_is_done(done)

            # self.agent.train_ppo()

        self.save_config()

        self.results.append((self.rank, total_episodes / self.cnf.main.n_steps))
        return self.results

    def save_config(self):
        if self.rank == 0:
            results_folder = "/home/julius/projects/curious/results/ms4/"
            with open(results_folder + "config.json", "w") as f:
                json.dump(OmegaConf.to_container(self.cnf, resolve=True), f)

    @staticmethod
    def plot(results):
        res = []
        for key, value in results.items():
            res.append(value[0])
        df = pd.DataFrame(res, columns=["Rank", "Episode length"])
        df.to_csv("results/ms6/without-im-easy.csv")
        print(df)
