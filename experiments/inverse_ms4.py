from .experiment import BaseExperiment
import altair as alt
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
import torch
import pickle
import json
import time


class Experiment(BaseExperiment):
    """
    Implements milestone 4 of the inverse experiments.

    Put the approach to test in a scenario with 2DOF and a
    static goal. The aim of this experiment is to show the
    benefits of the method vs. random exploration and do some
    hyperparameter tuning (4.1).

    Plot: Show the episode length of an agent with inverse actions
    vs. an agent with random actions. The agent with iv-actions should
    converge way faster since it has access to the simulated environment
    dynamics.
    """

    grid_size = 17

    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.results = []
        self.iv_state_template = f"out/inverse_state/rank{self.rank}_ms4_2dof_0.pt"

    def _gen_dataset(self):
        # first make the data set
        dataset = []
        state = self.env.reset()

        for i in range(100000):
            if i % 10000 == 0:
                print(f"rank {self.rank} at step", i)
            action = self.env.action_space.sample()
            next_state, *_ = self.env.step(action)
            dataset.append((state, next_state, action))
            state = next_state

        with open(f"out/ds/iv_gen_dataset_prop_rank{self.rank}_2dof.p", "wb") as f:
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
        if self.rank % 2 == 0:
            alpha = 0
            self.agent.set_alpha(alpha)
        else:
            alpha = self.cnf.ppo.alpha
        self.load_iv_state()
        # acquire goal
        print("generating goal")
        for i in range(100):
            goal, *_ = self.env.step([1] * self.cnf.env.action_dim)

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

                ep_len += 1
                if dist < 0.5:
                    done = True
                    reward = 10

                if ep_len > 200:
                    done = True
                    reward = 0

                reward_sum += reward
                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

            self.agent.train_ppo()
            if i % 1 == 0:
                self.results.append((self.rank, i, ep_len, alpha,))
            self.wandb.log({"episode_length": ep_len, "reward sum": reward_sum})

        self.save_config()

        return self.results

    def save_config(self):
        if self.rank == 0:
            results_folder = "/home/julius/projects/curious/results/ms4/"
            with open(results_folder + "config.json", "w") as f:
                json.dump(OmegaConf.to_container(self.cnf, resolve=True), f)

    @staticmethod
    def plot(results):
        results_folder = "/home/julius/projects/curious/results/ms4/"
        results_as_list = []
        for key, value_p in results.items():
            results_as_list += value_p

        df = pd.DataFrame(
            results_as_list, columns=["Rank", "Episode", "ep_len", "Alpha",],
        )

        df["rolling mean"] = (
            df.groupby("Rank")
            .rolling(15, min_periods=1)["ep_len"]
            .mean()
            .reset_index(level=0, drop=True)
        )

        df.to_csv(results_folder + time.strftime("%b %-d %H:%M:%S") + ".csv")
        base = alt.Chart(df)
        chart = base.mark_line().encode(
            x="Episode",
            y=alt.Y("mean(rolling mean):Q", title="Episode length"),
            color=alt.Color("Alpha:N"),
        )
        band = base.mark_errorband(extent="stdev").encode(
            x="Episode",
            y=alt.Y("rolling mean:Q", title="Episode length"),
            color=alt.Color("Alpha:N"),
        )
        (chart + band).show()

