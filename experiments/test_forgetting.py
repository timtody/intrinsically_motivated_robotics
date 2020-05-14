from .experiment import BaseExperiment
import numpy as np
import torch
import pickle


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
        self.episode_len = 250

        self.burnin_len = 1000000
        self.buffer_length = 250
        self.buffer_gap = 12

        self.train_len = 1000000
        self.train_every = 500

        # buffer for transitions
        self.trans_buffer = []

        self.global_step = 0

    def _burn_in(self):
        state = self.env.reset()
        print("starting burn-in")
        for i in range(self.burnin_len):

            if i % 10000 == 0:
                print("burn-in at step", i)

            action = self.agent.get_action(state)
            next_state, _, done, info = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, action)

            state = next_state

            if i % self.episode_len == self.episode_len - 1:
                done = True
                state = self.env.reset()

            self.agent.set_is_done(done)

            if i % self.train_every == self.train_every - 1:
                # print("training")
                results = self.agent.train()
                self.wandb.loss({"burn-in loss": results["imloss"].mean()}, step=i)

    def _gather_dataset(self):
        # collect data
        print("start gathering data...")
        state = self.env.reset()
        for i in range(self.buffer_length * self.buffer_gap):
            action = self.agent.get_action(state)
            next_state, _, done, info = self.env.step(action)

            if i % self.buffer_gap == 0:
                self.trans_buffer.append([state, next_state, action])

            state = next_state

    def _create_db(self):
        with open(f"rank_{self.rank}_database.p", "wb") as f:
            pickle.dump(self.trans_buffer, f)
        self.agent.save_state(f"rank_{self.rank}_")

    def _restore_exp(self):
        # restore saved database of transitions
        with open(f"data/forget/{self.rank}/database.p", "rb") as f:
            self.trans_buffer = pickle.load(f)
        # restore agent
        self.agent.load_state(f"data/forget/{self.rank}")

    def _test_forgetting(self):

        # start the exp
        self.agent.reset_buffers()
        state = self.env.reset()
        print("starting training")
        for i in range(self.train_len):
            self.global_step += 1
            action = self.agent.get_action(state)
            next_state, _, done, info = self.env.step(action)
            self.agent.append_icm_transition(state, next_state, action)

            if i % self.episode_len == self.episode_len - 1:
                done = True
                self.env.reset()

            self.agent.set_is_done(done)

            if i % self.train_every == self.train_every - 1:
                # print("training")
                # train the agent
                self.agent.train()
                # check how performance has changed after training
                state_batch, next_state_batch, action_batch = zip(*self.trans_buffer)
                loss = self.agent.icm.train_forward(
                    state_batch, next_state_batch, action_batch, freeze=True
                )
                self.wandb.log({"mean loss": loss.mean()}, step=self.global_step)

            state = next_state

    def run(self):
        self._burn_in()
        self._gather_dataset()
        self._create_db()
        self._test_forgetting()
