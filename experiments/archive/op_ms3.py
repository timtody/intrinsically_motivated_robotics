"""
Use a linear inverse model to learn the inverse problem on 7DOF data sets
"""
import os
import torch
import ctypes
import numpy as np
from .experiment import BaseExperiment
from multiprocessing import Array


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.cnf = cnf
        self.rank = rank
        self.target_folder = "out/ds/off-policy"
        self.iv_state_template = (
            f"out/ds/off-policy/iv-state-rank{self.rank}"
            + f"-{self.cnf.main.n_steps}-steps-im-{self.cnf.main.with_im}"
        )
        self.ep_len = 500
        self.episodes = 25

    def load_iv_state(self):
        print("Loading inverse state...")
        print(self.iv_state_template)
        self.agent.icm.load_inverse_state(self.iv_state_template)

    def compute_dist(self, state, goal):
        return ((goal - state) ** 2).mean()

    def generate_goals(self, easy=20, medium=20, hard=20):
        easy_range = [30, 60]
        med_range = [60, 120]
        hard_range = [120, 220]
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

            draw = np.random.randint(goal_range[0], high=goal_range[1])
            for _ in range(draw):
                goal, *_ = self.env.step(self.env.action_space.sample())
            goals.append(goal)
        return goals

    def _run(self):
        ep_lens = []
        # make goal
        self.env.reset()
        goals = self.generate_goals(easy=0, medium=0, hard=10)

        for i in range(self.episodes):
            goal = goals[np.random.randint(len(goals))]
            state = self.env.reset()
            for j in range(self.ep_len):
                action = self.agent.get_action(state, goal=goal)
                state, *_ = self.env.step(action)
                dist = self.compute_dist(state, goal)
                if dist < 5:
                    ep_lens.append(j)
                    break
                if j > 400:
                    ep_lens.append(j)
                    break
        return np.array(ep_lens)

    def run(self, pre_run_results):
        self.cnf.main.with_im = False
        self.load_iv_state()
        ep_lens_noim = self._run()
        self.cnf.main.with_im = True
        self.load_iv_state()
        ep_lens_im = self._run()
        print(
            "NOIM:",
            ep_lens_noim.mean(),
            ep_lens_noim.var(),
            "\n",
            "IM:",
            ep_lens_im.mean(),
            ep_lens_im.var(),
        )
