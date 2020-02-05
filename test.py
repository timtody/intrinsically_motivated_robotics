from env.environment import Env
from algo.ppo_cont import PPO, Memory
from utils import get_conf, prepare_wandb
from algo.models import ICModule
import numpy as np
import wandb

cnf = get_conf("conf/cnf_test.yaml")
env = Env(cnf)
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(action_dim, state_dim)

# setup target shape
target_position = env.get_target_position()


def get_reward(env):
    tip_position = env.get_tip_position()
    return -np.linalg.norm(np.array(target_position) - np.array(tip_position))


def get_done(reward):
    if reward > -0.15:
        return True
    return False


# prepare logging
prepare_wandb(cnf, agent, icmodule)

for i in range(1000):
    done = False
    obs = env.reset()
    timestep = 0
    for i in range(500):
        timestep += 1
        action = agent.policy_old.act(obs.get(), memory)
        next_obs, _, done, _ = env.step(
            [*action, *[0 for _ in range(7 - cnf.main.action_dim)]])
        reward = get_reward(env)
        done = get_done(reward)
        memory.rewards.append(reward)
        memory.is_terminals.append(done)

        if i % 100 == 99:
            loss, value_loss = agent.update(memory)
            if cnf.wandb.use:
                wandb.log({"loss": loss, "value_loss": value_loss})
            memory.clear_memory()

        if done:
            print("done")
            break

    if cnf.wandb.use:
        wandb.log({"is done": int(done), "episode len": timestep})
