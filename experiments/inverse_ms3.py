from .experiment import BaseExperiment
import altair as alt
import pandas as pd
import numpy as np
import torch
import pickle
import time


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
        self.results = []

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
                self.wandb.log({"eval loss": loss.mean()}, step=i)

            if i % 50 == 0:
                self.wandb.log({"training loss": loss.mean()}, step=i)

            # self.save_iv_state()

    def save_iv_state(self):
        self.agent.icm.save_inverse_state(
            f"out/inverse_state/rank{self.rank}_exp0_2dof.pt"
        )

    def compute_reward(self, state, goal):
        return ((goal - state) ** 2).sum()

    def compute_act_seq(self):
        """
        Computes the sequence of actions needed to check all
        positions around the agent in a square with side length
        Experiment.grid_size.
        """

        actions = []
        start_idx = Experiment.grid_size // 2
        for i in range(-start_idx, start_idx + 1):
            for j in range(-start_idx, start_idx + 1):
                actions += list(np.sign(i) * np.array(abs(i) * [[0, -1]]))
                actions += list(np.sign(j) * np.array(abs(j) * [[1, 0]]))
                actions += ["stop"]
        return actions

    def compute_matrix(self):
        goal = self.env.reset()
        action_sequence = self.compute_act_seq()

        results = np.zeros(Experiment.grid_size ** 2)
        i = 0
        for act in action_sequence:
            if act == "stop":
                dist_pre = self.compute_reward(state, goal)
                iv_action = self.agent.get_inverse_action(state, goal)
                self.env.step(iv_action * 0)
                self.env.step(iv_action * 0)
                state, *_ = self.env.step(iv_action)
                dist_post = self.compute_reward(state, goal)
                results[i] = -(dist_pre - dist_post)
                i += 1
                state = self.env.reset()
            else:
                state, *_ = self.env.step(1 * np.array(act))
        return results

    def run(self):
        train_set, test_set = self._load_dataset()
        self.train_inverse_model(train_set, test_set)
        results = self.compute_matrix()
        return results

    @staticmethod
    def plot(results):
        x, y = np.meshgrid(
            range(-Experiment.grid_size // 2 + 1, Experiment.grid_size // 2 + 1),
            range(-Experiment.grid_size // 2 + 1, Experiment.grid_size // 2 + 1),
        )
        print(x.shape)
        print(y.shape)
        print(results[0].reshape(x.shape[0], x.shape[0]))
        z = results[0]
        source = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "z": z})
        alt.Chart(source).mark_rect().encode(
            x="x:O", y="y:O", color=alt.Color("z:Q",)
        ).show()

