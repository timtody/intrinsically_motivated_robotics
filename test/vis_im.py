from utils import get_conf
from env.environment import Env
from algo.ppo_cont import PPO, Memory
from algo.models import ICModule
from utils import GraphWindow, Plotter3D
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape
import numpy as np

# get config setup
cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
# log = Logger.setup(cnf)
# init models
cnf.main.action_dim = 7
win = GraphWindow(["fs0 x", "fs0 y", "fs0 z", "fs1 x", "fs1 y", "fs1 z"],
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
scale = 0.06
obj_timer = 2
sound_sphere = None
# setup arm
for i in range(50000):
    timestep += 1
    action = agent.policy_old.act(state.get(), memory)
    # print(action)
    # next_state, *_, info = env.step(
    #     [*action, *[0 for _ in range(7 - cnf.main.action_dim)]])
    # next_state, *_, info = env.step([0, 1, 0, 0, 0, 0, 0])
    next_state, *_, info = env.step(env.action_space.sample())
    n_collisions += info["collided"]
    snd_intensity = next_state.audio[0]
    if snd_intensity > 0:
        print("bruh")
        theta = next_state.audio[2]
        phi = next_state.audio[3]
        head_pos = env.head.get_position()
        r = next_state.audio[1]
        x = r * np.sin(theta) * np.cos(phi) + head_pos[0]
        y = r * np.sin(theta) * np.sin(phi) + head_pos[1]
        z = r * np.cos(theta)  # + head_pos[2]

        sound_sphere = Shape.create(type=PrimitiveShape.SPHERE,
                                    color=[110, 110, 22],
                                    size=[
                                        snd_intensity * scale,
                                        snd_intensity * scale,
                                        snd_intensity * scale
                                    ],
                                    position=[x, y,  z],
                                    static=True,
                                    respondable=False)

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
# [3.7204288622169894, 1.127824017100505, 1.082173591098368, 0.9289968078727373]
# [3.720451520871281, 1.1284576367696812, 1.5623364866794647, 0.8678074238398878]
