from .experiment import BaseExperiment
import altair as alt
import pandas as pd
import numpy as np
import torch
import pickle
import seaborn as sns
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt


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

    grid_size = 29

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
        _state = self.env.reset()

        for i in range(self.cnf.main.iv_train_steps):
            idx = np.random.randint(len(train_set), size=1000)
            state_batch, next_state_batch, action_batch = zip(*train_set[idx])
            loss = self.agent.icm.train_inverse(
                state_batch,
                next_state_batch,
                torch.tensor(action_batch).to(self.device),
                eval=False,
            )

            if i % 500 == 0:
                print("At step", i)
            #     print("evaluating...")
            #     state_batch, next_state_batch, action_batch = zip(*test_set)
            #     loss = self.agent.icm.train_inverse(
            #         state_batch,
            #         next_state_batch,
            #         torch.tensor(action_batch),
            #         eval=True,
            #     )
            #     self.wandb.log({"eval loss": loss.mean()}, step=i)

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
        goal = state = self.env.reset()
        action_sequence = self.compute_act_seq()

        results = np.zeros(Experiment.grid_size ** 2)
        results_norm = np.zeros(Experiment.grid_size ** 2)
        i = 0
        for act in action_sequence:
            if act == "stop":
                dist_pre = self.compute_reward(state, goal)
                iv_action = self.agent.get_inverse_action(
                    torch.tensor(state).to(self.device),
                    torch.tensor(goal).to(self.device),
                )
                self.env.step(iv_action * 0)
                self.env.step(iv_action * 0)
                state, *_ = self.env.step(iv_action)
                dist_post = self.compute_reward(state, goal)
                delta = -(dist_pre - dist_post)
                if i == (Experiment.grid_size ** 2) // 2:
                    results[i] = dist_post + 1
                else:
                    results[i] = dist_post / dist_pre
                results_norm[i] = iv_action.norm()
                i += 1
                state = self.env.reset()
            else:
                state, *_ = self.env.step(1 * np.array(act))
        return results

    def run(self, pre):
        train_set, test_set = self._load_dataset()
        results_pre = self.compute_matrix()
        self.train_inverse_model(train_set, test_set)
        results_post = self.compute_matrix()
        return results_pre, results_post

    @staticmethod
    def plot(results):
        current_time = time.strftime("%b %-d %H:%M:%S")
        results_folder = "/home/julius/projects/curious/results/ms3/"
        run_name = "ticky-flanger"
        n_procs = 3
        print(results)

        # fig, axs = plt.subplots(ncols=n_procs, nrows=2, figsize=(16, 7))
        x, y = np.meshgrid(
            range(-Experiment.grid_size // 2 + 1, Experiment.grid_size // 2 + 1),
            range(-Experiment.grid_size // 2 + 1, Experiment.grid_size // 2 + 1),
        )

        vmin = np.inf
        vmax = -np.inf

        for _, v in results.items():
            vmin = min(vmin, v[0].min(), v[1].min())
            vmax = max(vmax, v[0].max(), v[1].max())

        sources = []

        def process_data(data):
            data_pre = pd.DataFrame({"x": x.ravel(), "y": y.ravel(), "dist": data})
            data_pre = data_pre.pivot("x", "y", "dist")
            return data_pre

        for i in range(n_procs):
            data_pre = process_data(results[i][0])
            data_post = process_data(results[i][1])
            data_pre.to_csv(f"results/ms3/bigresult_rank{i}_pre.csv")
            data_post.to_csv(f"results/ms3/bigresult_rank{i}_post.csv")
            # data_pre.to_csv(f"results/ms3/test_rnd{i+12}_post.csv")

        #     if i > 0:
        #         data_pre.index.name = " "
        #         data_post.index.name = " "
        #         yticklabels = False
        #     else:
        #         yticklabels = True

        #     sns.heatmap(
        #         data_pre,
        #         vmin=vmin,
        #         vmax=vmax,
        #         yticklabels=yticklabels,
        #         cbar=False,
        #         ax=axs[0][i],
        #         cmap="viridis_r",
        #     )
        #     sns.heatmap(
        #         data_post,
        #         vmin=vmin,to(self.device
        #         cbar=False,
        #         ax=axs[1][i],
        #         cmap="viridis_r",
        #     )

        # fig.tight_layout()
        # cb = fig.colorbar(axs[0][1].collections[0], ax=axs.ravel().tolist(),)
        # cb.set_label(
        #     label=r"$\frac{\Delta d(s_{x,y}, g)}{d(s_{x,y}, g)}$",
        #     weight="bold",
        #     size=16,
        # )

        # plt.savefig("test.pdf", bbox_inches="tight")
        # plt.show()
