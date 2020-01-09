import gym
import rlbench.gym
import wandb
import numpy as np
from omegaconf import OmegaConf
from models import ICModule
from ppo_cont import PPO, Memory
from wrappers import ObsWrapper

cnf = OmegaConf.load("conf/conf.yaml")
cnf.merge_with_cli()


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
    # wandb.watch(agent.policy, log="all")
    wandb.watch(icmodule, log="parameters")

state = env.reset()
done = False
timestep = 0
gripper_positions = []
for i in range(cnf.main.max_timesteps):
    timestep += 1
    gripper_positions.append(state.gripper_pose[:3])
    action = agent.policy_old.act(state.get_low_dim_data(), memory)
    next_state, _, done, _ = env.step(action)
    # env.render()
    im_loss = icmodule.train_forward(state.get_low_dim_data(),
                                     next_state.get_low_dim_data(),
                                     action)
    if cnf.wandb.use:
        wandb.log({
            "intrinsic reward": im_loss
        })
    # IM loss = reward currently
    reward = im_loss
    memory.rewards.append(reward)
    memory.is_terminals.append(done)
    state = next_state
    if timestep % cnf.main.train_each == 0:
        agent.update(memory)
        memory.clear_memory()
        timestep = 0
    if i % 500 == 499 and cnf.wandb.use:
        wandb.log({
            "gripper positions": wandb.Object3D(np.array(gripper_positions))
            })

env.close()
