import gym
import rlbench.gym
import wandb
import numpy as np
from omegaconf import OmegaConf
from models import ICModule
from ppo_cont import PPO, Memory
from wrappers import ObsWrapper
from utils import ColorGradient, PointCloud
import plotly.graph_objects as go


cnf = OmegaConf.load("conf/conf.yaml")
cnf.merge_with_cli()
# enable strict mode
OmegaConf.set_struct(cnf, True)

env = gym.make(cnf.main.env_name)
env = ObsWrapper(env)

# move to cnf file
obs_space = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(obs_space, action_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(obs_space, 1, action_dim)

if cnf.wandb.use:
    wandb.init(project=cnf.wandb.project, name=cnf.wandb.name)
    wandb.watch(agent.policy, log="all")
    wandb.watch(icmodule._forward, log="all")

state = env.reset()
done = False
timestep = 0
gripper_positions = []

# color gradients for visualizing point cloud
gradient = ColorGradient()

cum_im_reward = 0
for i in range(cnf.main.max_timesteps):
    timestep += 1
    gripper_positions.append(np.array([*state.gripper_pose[:3],
                                       100, 100, 100]))

    action = agent.policy_old.act(state.get_low_dim_data(), memory)
    next_state, _, done, _ = env.step(action)
    # env.render()
    im_loss = icmodule.train_forward(state.get_low_dim_data(),
                                     next_state.get_low_dim_data(),
                                     action)
    im_loss_processed = icmodule._process_loss(im_loss)
    cum_im_reward += im_loss_processed
    if cnf.wandb.use:
        wandb.log({
            "intrinsic reward raw": im_loss,
            "intrinsic reward": im_loss_processed,
            "return std": icmodule.loss_buffer.get_std(),
            "cummulative im reward": cum_im_reward,
            **{f"joint {i}": action[i] for i in range(len(action))}
        })
    # IM loss = reward currently
    reward = im_loss_processed
    memory.rewards.append(reward)
    memory.is_terminals.append(done)
    state = next_state
    last_action = action

    if timestep % cnf.main.train_each == 0:
        agent.update(memory)
        memory.clear_memory()
        timestep = 0
    if i % 500 == 499 and cnf.wandb.use:
        print(np.array(gripper_positions)[0])
        print(np.array(gripper_positions).shape)
        wandb.log({
            "gripper positions": wandb.Object3D(np.array(gripper_positions))
            })

point_cloud = PointCloud()
x, y, z = point_cloud.get_outer()
x0, y0, z0, r, g, b = zip(*gripper_positions)
fig = go.Figure(data=[
    go.Scatter3d(x=x, y=y, z=z,
                 mode='markers'),
    go.Scatter3d(x=x0, y=y0, z=z0)
])
fig.show()
r = np.zeros_like(x) + 100
g = np.zeros_like(y) + 100
b = np.zeros_like(z) + 100
outer_layer = np.array(list(zip(x, y, z, r, g, b)))
print(outer_layer[0])
print(outer_layer.shape)
wandb.log({
            "gripper positions": wandb.Object3D(outer_layer)
            })

env.close()
