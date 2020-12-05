"""
Creation of dataset with 10M transitions.
Use both IM and no IM
"""
import torch
from .experiment import BaseExperiment
import matplotlib.style as style
from inverse_model import IVModel
import numpy as np
import pandas as pd
from datetime import datetime

style.use("ggplot")


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)

    def compute_reward(self, state, goal):
        return -(np.linalg.norm(state - goal) ** 2) / 100

    def generate_goals(self, easy=20, medium=20, hard=20):
        easy_range = [5, 10]
        med_range = [10, 25]
        hard_range = [25, 35]
        goals = []
        goals += self._generate_goals(easy, easy_range)
        goals += self._generate_goals(medium, med_range)
        goals += self._generate_goals(hard, hard_range)
        return goals

    def _generate_goals(self, n, goal_range):
        goals = []
        for _ in range(n):
            self.env.reset()
            sign_draw = np.random.choice([-1, 1], size=7)
            horizontal_draw = np.random.randint(goal_range[0], high=goal_range[1])

            for i, sign in enumerate(sign_draw):
                action = [0] * self.cnf.env.action_dim
                action[i] = sign
                for _ in range(horizontal_draw):
                    goal, *_ = self.env.step(action)

            goals.append(goal)
        return goals

    def run(self, pre_run_results):
        goals = self.generate_goals(easy=0, medium=0, hard=1)
        goal = goals[0]

        model = IVModel(self.cnf, 2)
        if self.cnf.main.with_im:
            model.load_state_dict(
                torch.load(
                    "checkpoints/prod/ckpnt-50000-steps-2-layers-2020-12-02 21:13:06.133274"
                )
            )
        else:
            model.load_state_dict(
                torch.load("ckpnt-50000-steps-2-layers-noim-2020-12-04 16:57:56.069280")
            )

        self.agent.set_inverse_model(model)

        total_episodes = 0
        reward_sum = 0
        results = []

        for i in range(self.cnf.main.n_steps):
            if not self.rank:
                print("Rank 0 at episode", i)
            state = self.env.reset()
            # choose goal
            goal_idx = np.random.randint(len(goals))
            goal = goals[goal_idx]

            ep_len = 0
            done = False
            while not done:
                self.global_step += 1

                total_episodes += 1
                reward = 0

                action = self.agent.get_action(state, goal)
                next_state, *_ = self.env.step(action)
                dist = self.compute_reward(state, goal)

                ep_len += 1
                if dist > -0.05:
                    done = True
                    reward = 10

                if ep_len > 500:
                    done = True
                    reward = 0

                reward_sum += reward
                self.agent.add_transition(state, action, next_state, reward, done)

                if self.global_step > self.cnf.td3.burn_in:
                    self.agent.train()

                state = next_state

            results.append(
                (
                    self.rank,
                    goal_idx,
                    self.cnf.main.with_im,
                    i,
                    ep_len,
                    self.cnf.td3.alpha,
                )
            )

        return results

    @staticmethod
    def plot(results, cnf):
        res = []
        for key, value in results.items():
            for val in value:
                res.append(val)
        df = pd.DataFrame(
            res,
            columns=[
                "rank",
                "goal index",
                "with im",
                "step",
                "episode length",
                "alpha",
            ],
        )
        df.to_csv(f"results/icdl/all-diffs-im-{cnf.main.with_im}-{datetime.now()}.csv")
