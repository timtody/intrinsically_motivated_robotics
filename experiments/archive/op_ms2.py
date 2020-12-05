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
            + f"relu-im-{self.cnf.main.with_im}"
        )

    def load_iv_state(self):
        print("Loading inverse state...")
        print(self.iv_state_template)
        self.agent.icm.load_inverse_state(self.iv_state_template)

    def compute_dist(self, state, goal):
        return ((goal - state) ** 2).mean()

    def run(self, pre_run_results):
        self.load_iv_state()
        # make goal
        self.env.reset()
        for i in range(50):
            goal, *_ = self.env.step([0, 1, 0, 0, 0, 0, 0])

        state = self.env.reset()
        for i in range(100):
            action = self.agent.get_action(state, goal=goal)
            state, *_ = self.env.step(action)
            dist = self.compute_dist(state, goal)
            if dist < 5:
                print("Done in", i, "Steps.")
                exit(1)
