from utils import get_conf
from env.environment import Env
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from utils import GraphWindow

# get config setup
cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
# log = Logger.setup(cnf)
# init models
cnf.main.action_dim = 4
win = GraphWindow(2, 2, lookback=500)
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(action_dim, state_dim, **cnf.icm)

state = env.reset()
timestep = 0
while True:
    timestep += 1
    action = agent.policy_old.act(state.get(), memory)
    # print(action)
    next_state, *_ = env.step(
        [*action, *[0 for _ in range(7 - cnf.main.action_dim)]])
    reward_pre = icmodule.train_forward(state.get(), next_state.get(), action)
    im_reward = icmodule._process_loss(reward_pre)
    memory.rewards.append(im_reward)
    memory.is_terminals.append(False)
    win.update([
        im_reward, reward_pre, icmodule.loss_buffer.current_return,
        icmodule.loss_buffer.get_std()
    ])
    if timestep % 10 == -1:
        agent.update(memory)
        memory.clear_memory()
        timestep = 0
