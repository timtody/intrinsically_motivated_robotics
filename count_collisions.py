from env.environment import Env
from algo.models import ICModule
from conf import get_conf
from algo.ppo_cont import PPO, Memory
from utils import prepare_wandb
import numpy as np
import wandb

# get config setup
cnf = get_conf("conf/conf.yaml")
env = Env(cnf)

# init models
obs_space = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(obs_space, action_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(obs_space, 1, action_dim)

# prepare logging
prepare_wandb(cnf, agent, icmodule)

state = env.reset()
done = False
timestep = 0
gripper_positions = []

n_collisions = 0
timestep = 0
for _ in range(cnf.main.max_timesteps):
    timestep += 1
    action = agent.policy_old.act(state.get_all(), memory)
    next_state, _, done, info = env.step(action)
    im_loss = icmodule.train_forward(state.get_all(),
                                     next_state.get_all(),
                                     action)
    im_loss_processed = icmodule._process_loss(im_loss)
    memory.rewards.append(im_loss_processed)
    memory.is_terminals.append(done)
    state = next_state
    # count collisions (implicit coercion to int)
    n_collisions += info["collided_with_table"]
    # agent training
    if timestep % cnf.main.train_each == 0:
        agent.update(memory)
        memory.clear_memory()
        timestep = 0

if cnf.wandb.use:
    wandb.log({
        "n collisions": n_collisions
    })
print("Done. Number of collisions:", n_collisions)

env.close()
