from env.environment import Env
from algo.models import ICModule
from conf import get_conf
from algo.ppo_cont import PPO, Memory
from utils import prepare_wandb, PointCloud, Plotter3D
import pickle
import wandb
import plotly.graph_objects as go
import numpy as np

# get config setup
cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
# log = Logger.setup(cnf)
# init models
action_dim = env.action_space.shape[0]
action_dim = cnf.main.action_dim
state_dim = env.observation_space.shape[0]
agent = PPO(action_dim, state_dim, **cnf.ppo)
memory = Memory()
icmodule = ICModule(action_dim, state_dim)

# prepare logging
prepare_wandb(cnf, agent, icmodule)
"""
a = np.array([[2351, 236, 1090, 126, 8912], [6222, 7365,  120,  948, 5490], [5123, 3799,  3747, 1549, 1023]],)

"""
endresult_list = []

state_modes = ["notrain", "notouch", "all"]
for mode in state_modes:
    if mode == "notrain":
        cnf.main.train = False
    else:
        cnf.main.train = True
        cnf.env.state_size = mode
    trial_list = []
    for t in range(cnf.main.n_trials):
        # initialize new trial
        env.close()
        env = Env(cnf)
        action_dim = env.action_space.shape[0]
        action_dim = cnf.main.action_dim
        state_dim = env.observation_space.shape[0]
        agent = PPO(action_dim, state_dim, **cnf.ppo)
        memory = Memory()
        icmodule = ICModule(action_dim, state_dim)
        state = env.reset()
        done = False
        timestep = 0
        n_collisions = 0
        print(f"Starting mode {mode}, trial: {t}")
        for _ in range(cnf.main.max_timesteps):
            timestep += 1
            action = agent.policy_old.act(state.get(), memory)
            next_state, _, done, info = env.step(
                [*action, *[0 for _ in range(7 - cnf.main.action_dim)]])
            im_loss = icmodule.train_forward(state.get(), next_state.get(),
                                             action)
            im_loss_processed = icmodule._process_loss(im_loss)
            memory.rewards.append(im_loss_processed)
            memory.is_terminals.append(done)
            state = next_state
            # count collisions (implicit coercion to int)
            n_collisions += info["collided"]
            # agent training
            if timestep % cnf.main.train_each == 0 and cnf.main.train:
                agent.update(memory)
                memory.clear_memory()
                timestep = 0
        trial_list.append(n_collisions)
        # save the data

        if cnf.wandb.use:
            wandb.log({f"n collisions": n_collisions})
        print("Done. Number of collisions:", n_collisions)

    with open(f"data/{cnf.log.name}_cnf.log.name_{mode}", "wb") as f:
        pickle.dump(trial_list, f)

    endresult_list.append(trial_list)
env.close()

endresult_list = np.array(endresult_list)
print(endresult_list)

fig = go.Figure([
    go.Bar(x=state_modes,
           y=np.mean(endresult_list, axis=1),
           error_y=dict(type='data', array=np.std(endresult_list, axis=1)))
])
fig.write_html(f"data/{cnf.log.name}_result.html")
fig.show()
