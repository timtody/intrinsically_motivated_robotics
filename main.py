import gym
import rlbench.gym
import wandb
import numpy as np
from omegaconf import OmegaConf
from models import ICModule
from ppo_cont import PPO, Memory
from wrappers import ObsWrapper
from utils import ColorGradient, Plotter3D, PointCloud
from imageio import get_writer
import matplotlib.pyplot as plt
import torch


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
plotter3d = Plotter3D()
point_cloud = PointCloud()
plotter3d.plot_outer_cloud(point_cloud)

video_writer = get_writer(cnf.main.video_name +
                          str(0) + cnf.main.video_format,
                          fps=30)

cum_im_reward = 0
for i in range(cnf.main.max_timesteps):
    timestep += 1

    if i % cnf.main.video_length == cnf.main.video_length - 1:
        video_writer.close()
        video_writer = get_writer(cnf.main.video_name +
                                  str(i) + cnf.main.video_format,
                                  fps=30)

    gripper_positions.append(np.array(state.gripper_pose[:3]))
    action = agent.policy_old.act(state.get_low_dim_data(), memory)
    next_state, _, done, _ = env.step(action)
    output_img = env.render(mode="rgb_array")
    video_writer.append_data(output_img.copy())
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
    reward = im_loss_processed - torch.norm(action)
    memory.rewards.append(reward)
    memory.is_terminals.append(done)
    state = next_state
    last_action = action

    if timestep % cnf.main.train_each == 0:
        agent.update(memory)
        memory.clear_memory()
        timestep = 0
    if i % 500 == 499 and cnf.wandb.use:
        wandb.log({
            "gripper positions": wandb.Object3D(np.array(gripper_positions))
        })
    if i % 5000 == 4999:
        plotter3d = Plotter3D()
        plotter3d.plot_outer_cloud(point_cloud)
        plotter3d.plot_3d_data(gripper_positions)
        plotter3d.save(f"data/pc_step_{i}_{cnf.wandb.name}.html")

video_writer.close()

env.close()
