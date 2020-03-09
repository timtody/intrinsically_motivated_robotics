from env.environment import Env
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from torch.utils.tensorboard import SummaryWriter
from logger import Logger
from utils import get_conf, ReturnWindow

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
# win = GraphWindow(["reward", "reward raw", "return std", "value_fn"], 1, 4,
#                   10000)
# # tensorboard
writer = SummaryWriter()
window = ReturnWindow()

timestep = 0
state = env.reset()
i = 0
while True:
    i += 1
    value = 0
    timestep += 1
    action, _ = agent.policy_old.act(state.get(), memory)
    next_state, reward, done, _ = env.step(action.numpy())
    # compute im reward
    im_loss = icmodule.train_forward(state.get(), next_state.get(), action)
    im_loss_processed = icmodule._process_loss(im_loss)
    memory.is_terminals.append(done)
    memory.rewards.append(im_loss_processed)
    if timestep % cnf.main.train_each == 0:
        value = agent.policy_old.critic(memory.states[-1])
        ploss, vloss = agent.update(memory)
        memory.clear_memory()
        timestep = 0
        print("policy loss:", ploss, "\nvalue loss:", vloss)
    window.update(im_loss_processed.item(), agent.get_value(next_state.get()))

    state = next_state

    # for key, value in icmodule.base.named_parameters():
    #     writer.add_histogram(key, value, i)
    # writer.add_histogram("action mean", action_mean, i)
    # win.update(im_loss_processed, im_loss, icmodule.loss_buffer.get_std(),
    #            value)
