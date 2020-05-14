from .experiment import BaseExperiment
import plotly.graph_objects as go
import numpy as np
import torch
import time


class Experiment(BaseExperiment):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # setup logging metrics
        self.n_collisions_self = 0
        self.n_collisions_other = 0
        self.n_collisions_dynamic = 0
        self.n_sounds = 0
        self.reward_sum = 0

        # experiment parameters
        self.episode_len = 500
        self.episode_reward = 0

    def gather_states(self):
        state = self.env.reset()
        self.states = []
        for i in range(100):
            state, *_ = self.env.step(self.env.action_space.sample())
            if i % 20 == 0:
                self.states.append(state)

    def log_state_distances_baseline(self):
        distances = {}
        for i, state in enumerate(self.states):
            for j, other_state in enumerate(self.states):
                distances[f"baseline distance {i} {j}"] = self.state_dist(
                    state, other_state
                )
        self.wandb.log(distances)

    def log_state_distances(self, step):
        distances = {}
        for i, state in enumerate(self.states):
            for j, other_state in enumerate(self.states):
                distances[f"embedding distance {i} {j}"] = self.embed_dist(
                    state, other_state
                )
        self.wandb.log(distances, step=step)

    def state_dist(self, x, y):
        return ((x - y) ** 2).mean()

    def embed_dist(self, x, y):
        state = torch.tensor(x).float().to(self.device)
        other_state = torch.tensor(y).float().to(self.device)
        return (
            (self.agent.icm.embed(state) - self.agent.icm.embed(other_state)) ** 2
        ).mean()

    def run(self):
        self.gather_states()
        self.log_state_distances_baseline()
        state = self.env.reset()
        for i in range(self.cnf.main.n_steps):
            self.ppo_timestep += 1
            self.global_step += 1

            if not self.cnf.main.train:
                action = torch.tensor(self.env.action_space.sample())
            else:
                action = self.agent.get_action(state)

            next_state, _, done, info = self.env.step(action)

            self.agent.append_icm_transition(state, next_state, action)

            # reset environment
            if self.global_step % self.episode_len == 0:
                done = True
                # -------------
                self.env.reset()
                # TODO: change back
                # if self.cnf.main.train:
                #     self.env.reset()

            self.agent.set_is_done(done)

            # retrieve metrics
            self.n_collisions_self += info["collided_self"]
            self.n_collisions_other += info["collided_other"]
            self.n_collisions_dynamic += info["collided_dyn"]

            # TODO: reintroduce this
            # self.n_sounds += info["sound"]

            # train agent
            if self.ppo_timestep % self.cnf.main.train_each == 0:
                # train and log resulting metrics
                train_results = self.agent.train_with_inverse_reward()
                batch_reward = train_results["imloss"].sum().item()
                self.reward_sum += batch_reward

                # if we don't train we still want to log all the relevant data
                self.log_state_distances(self.global_step)
                self.wandb.log(
                    {
                        "n collisions self": self.n_collisions_self,
                        "n collisions other": self.n_collisions_other,
                        "n collisions dyn": self.n_collisions_dynamic,
                        "col rate self": self.n_collisions_self / self.global_step,
                        "col rate other": self.n_collisions_other / self.global_step,
                        "col rate dyn": self.n_collisions_dynamic / self.global_step,
                        # "n sounds": self.n_sounds,
                        "cum reward": self.reward_sum,
                        "batch reward": batch_reward,
                        "policy loss": train_results["ploss"],
                        "value loss": train_results["vloss"],
                    },
                    step=self.global_step,
                )
                joint_angles = []
                actions_norms = []

            state = next_state

        self.env.close()
