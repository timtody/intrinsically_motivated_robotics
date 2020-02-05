from env.environment import Env
from algo.models import ICModule
from conf import get_conf
from algo.ppo_cont import PPO, Memory
import pickle
import plotly.graph_objects as go
import numpy as np
import torch
from multiprocessing import Array, Process


def run(rank, cnf, mode, results):
    # random seed generation
    np.random.seed()
    torch.manual_seed(np.random.randint(999))

    if mode == "notrain":
        cnf.main.train = False
    else:
        cnf.main.train = True
        cnf.env.state_size = mode
    # re-init random seed per process
    env = Env(cnf)
    action_dim = env.action_space.shape[0]
    action_dim = cnf.main.action_dim
    state_dim = env.observation_space.shape[0]
    agent = PPO(action_dim, state_dim, **cnf.ppo)
    memory = Memory()
    icmodule = ICModule(action_dim, state_dim)
    state = env.reset()
    timestep = 0
    n_collisions = 0
    # start the experiment
    print(f"Starting mode {mode}, rank: {rank}")
    for i in range(cnf.main.max_timesteps):
        # some logging
        if (i + 1) % 1000 == 0:
            print(f"rank {rank} at step {i}.")
        timestep += 1
        action = agent.policy_old.act(state.get(), memory)
        next_state, _, done, info = env.step(
            [*action, *[0 for _ in range(7 - cnf.main.action_dim)]])
        im_loss = icmodule.train_forward(state.get(), next_state.get(), action)
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
    results[rank] = n_collisions
    env.close()


def run_mode_mp(mode, cnf):
    processes = []
    results = Array('d', range(cnf.mp.n_procs))
    for rank in range(cnf.mp.n_procs):
        p = Process(target=run, args=(rank, cnf, mode, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return results[:]


if __name__ == "__main__":
    # get config setup
    cnf = get_conf("conf/cnt_col.yaml")
    results = []
    state_modes = ["notrain", "notouch", "all"]
    for mode in state_modes:
        results.append(run_mode_mp(mode, cnf))
    results = np.array(results)
    print(results)
    fig = go.Figure([
        go.Bar(x=state_modes,
               y=np.mean(results, axis=1),
               error_y=dict(type='data', array=np.std(results, axis=1)))
    ])
    fig.write_html(f"data/{cnf.log.name}_result_mp.html")
    fig.show()
