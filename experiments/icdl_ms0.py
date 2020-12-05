"""
Creation of dataset with 10M transitions.
Use both IM and no IM
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from .experiment import BaseExperiment


class Experiment(BaseExperiment):
    def __init__(self, cnf, rank):
        super().__init__(cnf, rank)
        self.cnf = cnf
        self.rank = rank
        self.burn_in = 1000
        self.episode_len = 200
        self.global_step = 0

    def compute_reward(self, state, goal):
        return (np.linalg.norm(state - goal) ** 2) / 100

    def run(self, pre_run_results):
        state = self.env.reset()

        for i in range(30):
            goal, *_ = self.env.step([0.2, 1, 0.2, 0.2, 0.2, 0.2, 0.2])

        self.env.reset()
        state = self.env.reset()
        rewards = []

        for i in range(self.cnf.main.n_steps):

            state = self.env.reset()
            ep_reward = 0
            for j in range(self.episode_len):
                self.global_step += 1:
                if self.global_step < self.burn_in:
                    action = self.env.action_space.sample()
                else:
                    action = self.agent.get_action(state)

                next_state, reward, done, _ = self.env.step(action)
                reward = -self.compute_reward(next_state, goal)

                # they inverted the meaning of 'done'. YUCK!
                if reward > -0.05:
                    print("YES")
                    reward = 5
                    done = True
                else:
                    done = False
                ep_reward += reward
                self.agent.add_transition(state, action, next_state, reward, done)

                state = next_state

                if self.global_step > self.burn_in:
                    print("Training")
                    self.agent.train()

                if done:
                    break

            self.wandb.log({"episode reward": ep_reward})
