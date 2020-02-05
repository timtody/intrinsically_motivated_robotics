import gym
import rlbench.gym
from utils import get_conf
import wandb
from algo.ppo_cont import PPO, Memory

env = gym.make('reach_target-state-v0')
# Alternatively, for vision:
# env = gym.make('reach_target-vision-v0')
# get config setup
cnf = get_conf("conf/cnf_test.yaml")
# log = Logger.setup(cnf)
# init models
action_dim = env.action_space.shape[0]
observation_dim = env.observation_space.shape[0]
agent = PPO(action_dim, observation_dim, **cnf.ppo)
memory = Memory()

wandb.init(project="test", name="ppo-rlbench-gym")

# prepare logging
training_steps = 100000
episode_length = 100
timestep = 0
total_reward = 0

for i in range(training_steps):
    obs = env.reset()
    done = False
    for j in range(episode_length):
        timestep += 1
        action = agent.policy_old.act(obs, memory)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        memory.rewards.append(reward)
        memory.is_terminals.append(done)
        # env.render()  # Note: rendering increases step time.
        if timestep % 100 == 0:
            agent.update(memory)
            memory.clear_memory()
            timestep = 0
        if done:
            break

    wandb.log({"ep len": j, "total reward": total_reward})
    current_length = 0
    total_reward = 0
print('Done')
env.close()
