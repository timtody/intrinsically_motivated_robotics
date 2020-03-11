from env.environment import Env
import torch
import numpy as np
from utils import SkipWrapper, RewardQueue, ValueQueue
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
from abc import abstractmethod


class Experiment:
    def __init__(self, cnf, rank, mode):
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
        self.writer = SummaryWriter(f"tb/mode:{cnf.env.mode}_rank:{rank}")

        # set random seeds
        np.random.seed()
        torch.manual_seed(np.random.randint(9999))

    @abstractmethod
    def run(self, callbacks, log=False):
        pass


class CountCollisions(Experiment):
    def __init__(self, *args):
        super().__init__(*args)

        # setup logging metrics
        self.n_collisions = 0
        self.n_sounds = 0
        self.reward_sum = 0

        # queues
        self.Q_LEN = 50
        self.reward_Q = RewardQueue(self.Q_LEN, self.cnf.ppo.gamma)
        self.value_Q = ValueQueue(self.Q_LEN)

    def run(self, callbacks, log=False):
        state = self.env.reset()
        results = defaultdict(lambda: 0)

        for i in range(self.cnf.main.n_steps):
            self.ppo_timestep += 1
            if log and i % 5000 == 0:
                print("exp in mode", self.cnf.env.mode, "at step", i)

            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action, action_mean, entropy = self.agent.policy_old.act(
                    state.get(), self.memory)

            next_state, _, done, info = self.env.step(action)

            # calculate intrinsic reward
            im_loss = self.icm.train_forward(state.get(), next_state.get(),
                                             action)
            self.memory.rewards.append(im_loss)
            self.memory.is_terminals.append(done)

            # train agent
            if self.cnf.main.train:
                if self.ppo_timestep % self.cnf.main.train_each == 0:
                    ploss, vloss = self.agent.update(self.memory)
                    self.memory.clear_memory()
                    self.ppo_timestep = 0
                    self.writer.add_scalar("policy loss", ploss,
                                           self.global_step)
                    self.writer.add_scalar("value loss", vloss,
                                           self.global_step)

            # calculate mean / sum reward
            self.reward_sum += im_loss

            # calculate return / approx. return
            self.reward_Q.push(im_loss)
            self.value_Q.push(self.agent.get_value(next_state.get()))

            state = next_state

            # receive callback info
            for i, cb in enumerate(callbacks):
                results[i] += cb(info)

            # retrieve metrics
            self.n_collisions += info["collided"]
            self.n_sounds += info["sound"]

            # log to tensorboard
            if self.cnf.main.train:
                # training-only metrics
                self.writer.add_histogram("action_mean", action_mean,
                                          self.global_step)
                self.writer.add_scalar("entropy", entropy, self.global_step)

                # record action strengths
                for i in range(7):
                    self.writer.add_scalar(f"action_joint_{i}", action[i],
                                           self.global_step)

            # rest of metrics
            self.writer.add_scalar("reward", im_loss, self.global_step)
            self.writer.add_scalar("mean reward",
                                   self.reward_sum / self.global_step,
                                   self.global_step)
            self.writer.add_scalar("n_collisions", self.n_collisions,
                                   self.global_step)
            self.writer.add_scalar("n_sounds", self.n_sounds, self.global_step)

            # delayed return approximation
            if self.global_step >= self.Q_LEN:
                self.writer.add_scalars("ret_approx", {
                    "true_ret": self.reward_Q.get(),
                    "app_ret": self.value_Q.get()
                }, self.global_step - self.Q_LEN)

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


class CheckActor(Experiment):
    """ Experiment to investigate the 
    critic's function"""
    def run(self, callbacks, log=False):
        state = self.env.reset()
        results = defaultdict(lambda: 0)
        mean_reward = 0
        for i in range(self.cnf.main.n_steps):
            if log and i % 5000 == 0:
                print("exp in mode", self.cnf.env.mode, "at step", i)

            self.ppo_timestep += 1

            if not self.cnf.main.train:
                action = self.env.action_space.sample()
            else:
                action, action_mean, entropy = self.agent.policy_old.act(
                    state.get(), self.memory)

            next_state, _, done, info = self.env.step(action)

            if self.cnf.main.train:
                im_loss = self.icm.train_forward(state.get(), next_state.get(),
                                                 action)
                self.memory.rewards.append(im_loss)
                self.memory.is_terminals.append(done)
                mean_reward += im_loss
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
                self.writer.add_scalar("reward", im_loss, self.global_step)
                self.writer.add_scalar("mean reward",
                                       mean_reward / self.global_step,
                                       self.global_step)

            self.global_step += 1
        self.env.close()
        return results.values()
