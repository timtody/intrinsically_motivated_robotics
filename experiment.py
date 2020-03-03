import torch
import numpy as np
from utils import SkipWrapper
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from env.environment import Env


class Experiment:
    def __init__(self, cnf):
        self.cnf = cnf
        # setup env
        env = Env(cnf)
        skip_wrapper = SkipWrapper(cnf.main.skip)
        self.env = skip_wrapper(env)

        # setup agent
        action_dim = cnf.main.action_dim
        state_dim = env.observation_space.shape[0]
        self.agent = PPO(action_dim, state_dim, **cnf.ppo)
        self.memory = Memory()

        # setup ICM
        self.icm = ICModule(cnf.main.action_dim, state_dim, **cnf.icm)

        # setup experiment variables
        self.global_step = 0
        self.ppo_timestep = 0

        # set random seeds
        np.random.seed()
        torch.manual_seed(np.random.randint(9999))

    def run(self, callbacks, log=False):
        state = self.env.reset()
        results = {}
        for i in range(self.cnf.main.n_steps):
            if log and i % 5000:
                print("exp in mode", self.cnf.env.state_size, "at step", i)
            self.ppo_timestep += 1
            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action = self.agent.policy_old.act(state.get(), self.memory)

            next_state, _, done, info = self.env.step(action)

            if self.cnf.main.train:
                im_loss = self.icm.train_forward(state.get(), next_state.get(),
                                                 action)
                im_loss = self.icm._process_loss(im_loss)
                self.memory.rewards.append(im_loss)
                self.memory.is_terminals.append(done)
            state = next_state

            if self.cnf.main.train:
                if self.ppo_timestep % self.cnf.main.train_each == 0:
                    self.agent.update(self.memory)
                    self.memory.clear_memory()
                    self.ppo_timestep = 0

            # receive callback info
            for i, cb in enumerate(callbacks):
                results[i] = cb(info)
        self.env.close()
        return results.values()
