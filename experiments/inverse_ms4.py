from .experiment import BaseExperiment
import altair as alt
import pandas as pd
import numpy as np
import torch
import pickle


class Experiment(BaseExperiment):
    """
    Implements milestone 3 of the inverse experiments.

    Test how the inverse model generalizes by supplying a range
    of states around the goal state and inspecting the suggested
    actions. Do these actions take the agent closer towards the
    goal or are they just noise?

    Plot: Show heatmap of values which indicate the quality of
    the prediction i.e. a good prediction would be an action which
    took the agent closter to the goal than it is currently.

    """

    grid_size = 17

    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.results_iv_train = []
        self.results_p_train = []

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

            if i % 500 == 0:
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

            self.save_iv_state()

    def save_iv_state(self):
        self.agent.icm.save_inverse_state(
            f"out/inverse_state/rank{self.rank}_ms4_2dof.pt"
        )

    def load_iv_state(self):
        self.agent.icm.load_inverse_state(
            f"out/inverse_state/rank{self.rank}_ms4_2dof.pt"
        )

    def compute_dist(self, state, goal):
        return ((goal - state) ** 2).mean()

    def run(self):
        self.train_inverse_model()
        # acquire goal
        for i in range(100):
            goal, *_ = self.env.step([1] * self.cnf.env.action_dim)

        for i in range(self.cnf.main.n_steps):
            state = self.env.reset()

            ep_len = 0
            reward_proxy = 0
            done = False
            while not done:
                reward = -0.05

                action = self.agent.get_action(state, goal=goal)
                state, *_ = self.env.step(action)
                dist = self.compute_dist(state, goal)

                reward_proxy -= dist
                ep_len += 1

                if dist < 0.5:
                    done = True
                    reward = 10

                if ep_len > 100:
                    done = True
                    reward = 0

                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

            self.agent.train_ppo()
            self.results_p_train.append((self.rank, i, ep_len,))
            self.wandb.log({"episode_length": ep_len, "reward proxy": reward_proxy})

        return ()

    @staticmethod
    def plot(results):
        results_folder = "/home/julius/projects/curious/results/ms3/"
        with open(results_folder + "config.yaml", "w") as f:
            pickle.dump(Experiment.cnf, f)
