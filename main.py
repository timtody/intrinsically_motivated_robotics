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

if cnf.wandb.use:
    wandb.init(project=cnf.wandb.project, name=cnf.wandb.name)
    wandb.watch(agent.policy, log="all")

state = env.reset()
print(state[:16])
state, _, _, _ = env.step([1, 0, 0, 0, 0, 0, 0, 0])
for i in range(100):
    env.step([1, 0, 0, 0, 0, 0, 0, 0])
    print(state[:16])
exit(1)
done = False
timestep = 0
for i in range(cnf.main.max_timesteps):
    timestep += 1
    action = agent.policy_old.act(state, memory)
    next_state, _, done, _ = env.step(action)
    # env.render()
    loss = icmodule.train_forward(state, next_state, action)
    if cnf.wandb.use:
        wandb.log({
            "intrinsic reward": loss
        })
    # IM loss = reward currently
    reward = loss
    memory.rewards.append(reward)
    memory.is_terminals.append(done)
    state = next_state
    if timestep % 100 == 0:
        agent.update(memory)
        memory.clear_memory()
        timestep = 0
