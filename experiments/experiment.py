import os
import torch
import pickle
import time
import json
import wandb
import collections
import numpy as np
from observation import Observation
from environment import Env
from agent import Agent
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from collections import defaultdict
from abc import abstractmethod


class BaseExperiment:
    def __init__(self, cnf, rank):
        self.cnf = cnf
        self.rank = rank

        # saving and loading
        if self.cnf.main.checkpoint:
            self.create_checkpoint_path()
            self.create_checkpoint_dirs()

            checkpoint = self.cnf.main.checkpoint
            self.cnf = self.load_conf_from_checkpoint()
            self.cnf.main.checkpoint = checkpoint

        if self.cnf.main.save_state and not self.cnf.main.checkpoint:
            self.save_conf_to_checkpoint()

        # set random seeds
        np.random.seed(cnf.env.np_seed)
        torch.manual_seed(cnf.env.torch_seed)

        # setup env
        self.env = Env(cnf)

        # pytorch device
        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available() and cnf.main.gpu
            else torch.device("cpu")
        )
        # setup agent
        self.action_dim = cnf.env.action_dim
        self.state_dim = self.env.observation_space.shape[0]

        self.init_agent()

        # setup experiment variables
        self.global_step = 0
        self.ppo_timestep = 0

        self.init_wandb()

    def init_agent(self, is_goal_based=False):
        self.agent = Agent(
            self.action_dim,
            self.state_dim,
            self.cnf,
            self.device,
            is_goal_based=is_goal_based,
        )

    def init_wandb(self):
        # actually dont need to bind this here
        self.wandb = wandb

        self.wandb.init(
            config=self.cnf,
            project=self.cnf.wandb.project,
            name=f"{self.cnf.wandb.name}_rank{self.rank}",
            group=f"{self.cnf.wandb.group}",
            resume=bool(self.cnf.main.checkpoint),
        )
        wandb.watch(self.agent.ppo.policy)

    def create_checkpoint_dirs(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def create_checkpoint_path(self):
        # TODO: make this thread safe by moving it up in run.py
        self.checkpoint_path = os.path.join(
            "out", self.cnf.wandb.name, time.strftime("%B-%d:%H-%M-%S"),
        )

    def load_conf_from_checkpoint(self):
        return torch.load(os.path.join(self.cnf.main.checkpoint, "cnf.p"))

    def save_conf_to_checkpoint(self):
        torch.save(self.cnf, os.path.join(self.checkpoint_path, "cnf.p"))

    @abstractmethod
    def pre_run_hook(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def save_state(self, step, additional_state={}):
        path = os.path.join("checkpoints", str(step))
        if not os.path.exists(path):
            os.mkdir(path)
        self._save_extra(step, additional_state)
        self._save(step)
        self.agent.save_state(step)
        self.env.save_state(step)

    def load_state(self, checkpoint, load_env=False) -> dict:
        extra_state = {}  # self._load_extra(checkpoint)
        self._load(checkpoint)
        self.agent.load_state(checkpoint)
        self.env.load_state(checkpoint)
        return extra_state

    def _save_extra(self, step, state) -> None:
        save_path = os.path.join("checkpoints", str(step), "extra_state.p")
        with open(save_path, "wb") as f:
            pickle.dump(state, f)

    def _load_extra(self, checkpoint) -> dict:
        abspath = os.environ["owd"]
        load_path = os.path.join(abspath, checkpoint, "exp_state.json")
        with open(load_path, "rb") as f:
            extra_state = pickle.load(f)
        return extra_state

    def _save(self, step) -> None:
        save_path = os.path.join("checkpoints", str(step), "exp_state.json")
        print("saving exp at", save_path)
        state = dict(
            global_step=self.global_step,
            ppo_step=self.ppo_timestep,
            icm_buffer=self.icm_buffer,
            torch_seed=self.cnf.env.torch_seed,
            np_seed=self.cnf.env.np_seed,
        )
        with open(save_path, "w") as f:
            json.dump(state, f)

    def _load(self, checkpoint) -> None:
        # restores the state of the experiment
        abspath = os.environ["owd"]
        load_path = os.path.join(abspath, checkpoint, "exp_state.json")
        print("loading exp state from", load_path)
        with open(load_path, "r") as f:
            state = json.load(f)
        self.global_step = state["global_step"]
        self.ppo_timestep = state["ppo_step"]
        self.icm_buffer = state["icm_buffer"]
        # set random seeds
        torch.manual_seed(state["torch_seed"])
        np.random.seed(state["np_seed"])

    @staticmethod
    def plot(results, cnf):
        pass
