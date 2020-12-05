"""
Creation of dataset with 10M transitions.
Use both IM and no IM
"""
import torch
import matplotlib.pyplot as plt
from .experiment import BaseExperiment
from evaluate import IVEvaluator
import seaborn as sns
import matplotlib.style as style
from inverse_model import IVModel
import numpy as np
from datetime import datetime

style.use("ggplot")


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.iv_evaluator = IVEvaluator()

    def split_dataset(self, dataset):
        print("splitting dataset")
        split = 0.99
        test_set = dataset[int(len(dataset) * split) :]
        train_set = dataset[: int(len(dataset) * split)]

        print("successfully loaded and splitted dataset of length", len(dataset))
        return train_set, test_set

    def train_inverse_model(self, model, train_set, test_set, nlayers=4):
        print("rank", self.rank, "Training inverse model")
        for i in range(self.cnf.main.n_steps):
            idx = np.random.randint(len(train_set), size=self.cnf.main.bsize)
            states, nstates, actions = zip(*train_set[idx])
            loss = model.train(states, nstates, actions)

            if i % 1000 == 0:
                print("Rank", self.rank, "evaluating at global step", i)
                states, nstates, actions = zip(*test_set)
                loss = model.train(states, nstates, actions, eval=True)
                # self.wandb.log({f"im: {self.cnf.main.with_im} eval loss": loss}, step=i)

            if i % 50 == 0:
                pass
                # self.wandb.log(
                #     {f"im: {self.cnf.main.with_im} training loss": loss}, step=i
                # )
        torch.save(
            model.state_dict(),
            f"checkpoints/ckpnt-{self.cnf.main.n_steps}-steps-{model.depth}-layers-noim-{datetime.now()}",
        )

    def compute_reward(self, state, goal):
        print(state)
        print(goal)
        return -(np.linalg.norm(state - goal) ** 2) / 100

    def make_barplot(self, ax, results, name):
        mean = results.mean(axis=0)
        std = results.std(axis=0)

        ax.bar(
            range(1, len(mean) + 1), mean, yerr=std, color="#B1483B",
        )
        ax.set_ylabel(r"$E(\tau_i, g)$")
        ax.set_xlabel("i")
        ax.set_ylim((0, 3))
        ax.title.set_text(name)

    def eval_model(self, model, goals):
        device = torch.device("cuda")

        total_goals = len(goals)
        total_reward = 0
        for goal in goals:
            state = self.env.reset()
            reward = 0
            for i in range(100):
                action = (
                    model(
                        torch.tensor(state).float().to(device),
                        torch.tensor(goal).float().to(device),
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )
                next_state, *_ = self.env.step(action)
                reward += self.compute_reward(next_state, goal)
                print(reward)
                state = next_state
            # print("Toral episode reward:", reward)
            total_reward += reward
        # print("Mean reward:", total_reward / total_goals)
        return total_reward / total_goals

    def run(self, pre_run_results):
        ds = torch.load("out/ds/off-policy/dataset_without_im")
        train, test = self.split_dataset(ds)
        # model_1layer = IVModel(self.cnf, 1)
        model_2layer = IVModel(self.cnf, 2)
        # model_3layer = IVModel(self.cnf, 3)
        # model_4layer = IVModel(self.cnf, 4)

        # taus, goals = torch.load("easy_goals.p")

        # self.train_inverse_model(model_1layer, train, test)
        # rew1 = self.eval_model(model_1layer, goals)

        self.train_inverse_model(model_2layer, train, test)
        # rew2 = self.eval_model(model_2layer, goals)

        exit(1)

        self.train_inverse_model(model_3layer, train, test)
        rew3 = self.eval_model(model_3layer, goals)

        self.train_inverse_model(model_4layer, train, test)
        rew4 = self.eval_model(model_4layer, goals)

        print("total reward 1 layer:", rew1)
        print("total reward 2 layer:", rew2)
        print("total reward 3 layer:", rew3)
        print("total reward 4 layer:", rew4)

        exit(1)

        results_1layer = self.iv_evaluator.evaluate(model_1layer, taus, goals)
        # results_2layer = self.iv_evaluator.evaluate(model_2layer, taus, goals)
        # results_3layer = self.iv_evaluator.evaluate(model_3layer, taus, goals)
        # results_4layer = self.iv_evaluator.evaluate(model_4layer, taus, goals)

        fig, axes = plt.subplots(ncols=1, nrows=1)
        # axes = axes.flatten()
        self.make_barplot(axes, results_1layer, "Linear")
        # self.make_barplot(axes[1], results_2layer, "2 Layer")
        # self.make_barplot(axes[2], results_3layer, "3 Layer")
        # self.make_barplot(axes[3], results_4layer, "4 Layer")

        plt.tight_layout()
        plt.savefig("results/icdl/linear_after_training.pdf")
        plt.savefig("results/icdl/linear_after_training.png")
        plt.show()

