from .experiment import BaseExperiment
import altair as alt
import pandas as pd
import numpy as np
import torch
import pickle


class Experiment(BaseExperiment):
    """
    Implements milestone 2 of the inverse experiments.

    Test with static goal and 1 goal dimension.
    Show that this actually works under realistic scenarios
    i.e. small alpha of ~ 0.15 and a learned inverse model.
    Should yield results faster than random exploration.

    Plot: Episode length for random exploration vs with
    inverse action. Could be done by comparing alpha=0.15 with
    alpha=0.0.

    """

    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.results_iv_train = []
        self.results_p_train = []

    def _load_dataset(self):
        with open("out/dataset2_2dof.p", "rb") as f:
            dataset = np.array(pickle.load(f))

        split = 0.95
        dataset = np.array(dataset)
        np.random.shuffle(dataset)
        test_set = dataset[int(len(dataset) * split) :]
        train_set = dataset[: int(len(dataset) * split)]

        print("successfully loaded dataset of length", len(dataset))
        return train_set, test_set

    def train_inverse_model(self, train_set, test_set):
        print("Training inverse model")
        state = self.env.reset()

        for i in range(self.cnf.main.iv_train_steps):
            idx = np.random.randint(len(train_set), size=500)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.icm.train_inverse(
                state_batch, next_state_batch, torch.tensor(action_batch), eval=False
            )

            if i % 500 == 0:
                print("evaluating...")
                state_batch, next_state_batch, action_batch = zip(*test_set)
                loss = self.agent.icm.train_inverse(
                    state_batch, next_state_batch, torch.tensor(action_batch), eval=True
                )
                self.results_iv_train.append((self.rank, i, loss.mean().item(), "eval"))
                self.wandb.log({"eval loss": loss.mean()}, step=i)

            if i % 50 == 0:
                self.results_iv_train.append(
                    (self.rank, i, loss.mean().item(), "train")
                )
                self.wandb.log({"training loss": loss.mean()}, step=i)

        self.agent.icm.save_inverse_state(
            f"out/inverse_state/rank{self.rank}_exp0_2dof.pt"
        )

    def compute_reward(self, goal, state):
        return -((goal - state) ** 2).sum()

    def run(self):
        train_set, test_set = self._load_dataset()
        self.train_inverse_model(train_set, test_set)
        exit(1)

        self.env.reset()
        proto_action = [1] * self.cnf.env.action_dim

        # acquire goal
        for i in range(100):
            goal, *_ = self.env.step(proto_action)

        for i in range(self.cnf.main.n_steps):
            state = self.env.reset()
            done = False
            ep_len = 0

            while not done:
                ep_len += 1
                action = self.agent.get_action(state, goal=goal)
                state, *_ = self.env.step(action)
                distance = self.compute_reward(state, goal)
                reward = 0
                if distance > -5:
                    done = True
                    reward = 10

                if ep_len > 100:
                    done = True
                    reward = 0

                self.agent.set_reward(reward)
                self.agent.set_is_done(done)

            self.agent.train_ppo()
            self.results_p_train.append(
                (self.rank, i, ep_len, self.alpha_on, self.alpha_off,)
            )

        return (self.results_iv_train, self.results_p_train)

    @staticmethod
    def plot(results):
        pass
