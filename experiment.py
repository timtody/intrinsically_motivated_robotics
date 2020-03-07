from env.environment import Env
import torch
import numpy as np
from utils import SkipWrapper
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from abc import abstractmethod


class Experiment:
    def __init__(self, cnf, rank=0, mode=None):
        self.cnf = cnf
        # setup env
        env = Env(cnf)
        skip_wrapper = SkipWrapper(cnf.env.skip)
        self.env = skip_wrapper(env)

        # setup agent
        action_dim = cnf.env.action_dim
        state_dim = env.observation_space.shape[0]
        self.agent = PPO(action_dim, state_dim, **cnf.ppo)
        self.memory = Memory()

        # setup ICM
        self.icm = ICModule(cnf.env.action_dim, state_dim, **cnf.icm)

        # setup experiment variables
        self.global_step = 0
        self.ppo_timestep = 0

        # setup tensorboard
        self.writer = SummaryWriter(f"tb/rank:{rank}_mode:{mode}")

        # setup logging metrics
        self.n_collisions = 0

        # set random seeds
        np.random.seed()
        torch.manual_seed(np.random.randint(9999))

    @abstractmethod
    def run(self, callbacks, log=False):o
        pass


class CountCollisions(Experiment):
    def run(self, callbacks, log=False):
        state = self.env.reset()
        results = defaultdict(lambda: 0)
        for i in range(self.cnf.main.n_steps):
            if log and i % 5000 == 0:
                print("exp in mode", self.cnf.env.o, "at step", i)

            self.ppo_timestep += 1

            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action, action_mean = self.agent.policy_old.act(
                    state.get(), self.memory)

            next_state, _, done, info = self.env.step(action)

            if self.cnf.main.train:
                im_loss_pre = self.icm.train_forward(state.get(),
                                                     next_state.get(), action)
                im_loss = self.icm._process_loss(im_loss_pre)
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
                results[i] += cb(info)

            # retrieve metrics
            self.n_collisions += info["collided"]

            # log to tensorboard
            if self.cnf.main.train:
                self.writer.add_histogram("action_mean", action_mean,
                                          self.global_step)
                self.writer.add_scalar("reward", im_loss, self.global_step)
                self.writer.add_scalar("reward raw", im_loss_pre,
                                       self.global_step)
                self.writer.add_scalar("return std",
                                       self.icm.running_return_std,
                                       self.global_step)
            self.writer.add_scalar("n_collisions", self.n_collisions,
                                   self.global_step)

            self.global_step += 1
        self.env.close()
        return results.values()


class GoalReach(Experiment):
    def __init__(self):
        super().__init__()
        # init goal buffer here

    def run(self, callbacks, log=False):
        # run exp here
        pass
