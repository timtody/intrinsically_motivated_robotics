from utils import get_conf
from env.environment import Env
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from utils import GraphWindow, Plotter3D

# get config setup
cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
# log = Logger.setup(cnf)
# init models
cnf.main.action_dim = 7
win = GraphWindow(["fs0 x",
                   "fs0 y",
                   "fs0 z",
                   "fs1 x",
                   "fs1 y",
                   "fs1 z"],
                  3,
                  2,
                  lookback=500)
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(action_dim, state_dim, **cnf.icm)

state = env.reset()
timestep = 0
n_collisions = 0
plotter = Plotter3D()
gripper_positions = []
for i in range(50000):
    timestep += 1
    action = agent.policy_old.act(state.get(), memory)
    # print(action)
    # next_state, *_, info = env.step(
    #     [*action, *[0 for _ in range(7 - cnf.main.action_dim)]])
    next_state, *_, info = env.step([0, 1, 0, 0, 0, 0.5, 0])
    n_collisions += info["collided"]
    print(info["gripper_speed"])
    gripper_positions.append(env.get_tip_position())
    reward_pre = icmodule.train_forward(state.get(), next_state.get(), action)
    im_reward = icmodule._process_loss(reward_pre)
    # memory.rewards.append(im_reward)
    # memory.is_terminals.append(False)
    state = next_state
    # win.update(env.read_force_sensors_flat()[:6])
    # if timestep % 100 == -1:
    #     agent.update(memory)
    #     memory.clear_memory()
    #     timestep = 0
plotter.plot_3d_data(gripper_positions)
plotter.show()
plotter.save("data/point-cloud-no-im-sample.html")
print("noim", n_collisions)


# 173
# 35