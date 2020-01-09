import gym
import rlbench.gym
import wandb
from omegaconf import OmegaConf
from models import ICModule
from ppo_cont import PPO, Memory

cnf = OmegaConf.load("conf/conf.yaml")
cnf.merge_with_cli()

env = gym.make(cnf.main.env_name)

# move to cnf file
obs_space = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = PPO(obs_space, action_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(obs_space, 1, action_dim)

if cnf.main.use_wb:
    wandb.init(project=cnf.wandb.project, name=cnf.wandb.name)
    wandb.watch(agent.policy, log="all")

state = env.reset()
done = False
for i in range(cnf.main.max_timesteps):
    action = agent.policy_old.act(state, memory)
    print(action)
    exit(1)
    next_state, _, done, _ = env.step(action)
    loss = icmodule.train_forward(state, next_state, action)
    # IM loss = reward currently
    reward = loss
    agent.append_reward(reward)
    state = next_state
