from env.environment import Env
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from torch.utils.tensorboard import SummaryWriter
from logger import Logger
from utils import get_conf, ReturnWindow, GraphWindow
import numpy as np


class RewardQueue:
    def __init__(self, length, gamma):
        self.Q = np.zeros(length)
        self.len = length
        self.gamma = gamma

    def push(self, reward) -> None:
        self.Q = np.roll(self.Q, -1)
        self.Q[-1] = 0

        for i, r in enumerate(range(self.len - 1, -1, -1)):
            self.Q[i] += self.gamma**r * reward

    def get(self) -> int:
        return self.Q[0]


class ValueQueue:
    def __init__(self, length):
        self.len = length
        self.Q = np.zeros(length)

    def push(self, elem) -> None:
        self.Q[-1] = elem
        self.Q = np.roll(self.Q, -1)

    def get(self) -> int:
        return self.Q[0]


Q_LEN = 200
ret_Q = RewardQueue(Q_LEN, 0.5)
val_Q = ValueQueue(Q_LEN)

cnf = get_conf("conf/main.yaml")
# init env before logger!
# log = Logger(cnf)
env = Env(cnf)

# skip_wrapper = SkipWrapper(1)
# env = skip_wrapper(env)
action_dim = env.action_space.shape[0]
action_dim = cnf.env.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(action_dim, state_dim, **cnf.icm)
graph_win = GraphWindow(["app. ret"], 1, 1, lookback=10000)
# win = GraphWindow(["reward", "reward raw", "return std", "value_fn"], 1, 4,
#                   10000)
# # tensorboard
writer = SummaryWriter("tb/test_rew_q")
window = ReturnWindow(discount_factor=0.5, lookback=10000)

timestep = 0
ret_sum = 0
state = env.reset(random=True)
i = 0
running_ret = 0
return_window_size = 200
ret_window = np.empty(200)
while True:
    i += 1
    value = 0
    timestep += 1
    action, _ = agent.policy_old.act(state.get(), memory)
    next_state, reward, done, _ = env.step(action.numpy())
    # compute im reward
    im_loss = icmodule.train_forward(state.get(), next_state.get(), action)
    # im_loss_processed = icmodule._process_loss(im_loss)
    memory.is_terminals.append(done)
    memory.rewards.append(im_loss)

    ret_Q.push(im_loss)
    val_Q.push(agent.get_value(next_state.get()))
    if i >= Q_LEN:
        writer.add_scalars("return approximation", {
            "approx_ret": val_Q.get(),
            "true_ret": ret_Q.get()
        }, i - Q_LEN)

    if timestep % cnf.main.train_each == 0:
        value = agent.policy_old.critic(memory.states[-1])
        ploss, vloss = agent.update(memory)
        memory.clear_memory()
        timestep = 0
        print("policy loss:", ploss, "\nvalue loss:", vloss)

    window.update(im_loss.item(), agent.get_value(next_state.get()))

    ret_sum += im_loss
    # graph_win.update(im_loss, ret_sum / i)

    state = next_state
    # writer.add_scalar("loss", im_loss, i)
    # writer.add_scalar("mean reward", ret_sum / i, i)

    # if i % 250 == 0:
    #     env.reset()

    # for key, value in icmodule.base.named_parameters():
    #     writer.add_histogram(key, value, i)
    # writer.add_histogram("action mean", action_mean, i)
    # win.update(im_loss_processed, im_loss, icmodule.loss_buffer.get_std(),
    #            value)
