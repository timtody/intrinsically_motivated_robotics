import torch
import collections

from algo.ppo_cont import PPO, Memory
from algo.models import ICModule


class Agent:
    def __init__(self, cnf):
        # PPO related stuff
        self.ppo = PPO()
        self.ppo_mem = Memory()

        # ICM related stuff
        self.icm = ICModule()
        self.icm_transition = collections.namedtuple(
            "icm_trans", ["state", "next_state", "action"]
        )
        self.icm_buffer = []

    def append_icm_transition(self, this_state, next_state, action):
        self.icm_buffer.append(self.icm_transition(this_state, next_state, action))

    def set_is_done(self, is_done):
        self.ppo_mem.is_terminals.append(is_done)

    def get_action(self, state) -> torch.tensor:
        action, *_ = self.ppo.policy_old.act(state.get(), self.ppo_mem)
        return action

    def train(self) -> None:
        # TODO: return results dict
        # train icm
        state_batch, next_state_batch, action_batch = zip(*self.icm_buffer)
        im_loss_batch = self.icm.train_forward(
            state_batch, next_state_batch_action_batch
        )
        # train actor
        self.ppo_mem.rewards = im_loss_batch.cpu().numpy()
        self.icm_buffer = []
        self.ppo.update(self.ppo_mem)
        self.ppo_mem.clear_memory()

    def save_state(self, path) -> None:
        # save icm
        self.icm.save(path)
        # save ppo
        self.ppo.save(path)

    def load_state(self, path) -> None:
        # load icm
        self.icm.load(path)
        # load ppo
        self.ppo.load(path)
