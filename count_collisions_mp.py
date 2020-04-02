from experiment import CountCollisions
import pickle
import plotly.graph_objects as go
import numpy as np
from conf import get_conf
from multiprocessing import Array, Process
from logger import Logger
import multiprocessing as mp


def run(rank, cnf, mode, results):
    if mode == "notrain":
        cnf.main.train = False
    else:
        cnf.main.train = True
        cnf.env.state_size = mode
    cnf.env.mode = mode
    exp = CountCollisions(cnf, rank, log=True if rank == 0 else False)
    n_collisions = 0
    # start the experiment
    if rank == 0:
        print("Starting mode", mode)
    n_collisions, n_sounds = exp.run([lambda x: x["collided"], lambda x: x["sound"]])
    results[rank] = n_collisions


def run_mode_mp(mode, cnf):
    processes = []
    results = Array("d", range(cnf.mp.n_procs))
    for rank in range(cnf.mp.n_procs):
        p = Process(target=run, args=(rank, cnf, mode, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    return results[:]


if __name__ == "__main__":
    # get config setup
    mp.set_start_method("spawn")
    cnf = get_conf("conf/main.yaml")
    log = Logger(cnf)
    results = []
    state_modes = ["all"]
    for mode in state_modes:
        results.append(run_mode_mp(mode, cnf))
    results = np.array(results)
    with open(f"data/n_collisions.p", "wb") as f:
        pickle.dump(results, f)
    fig = go.Figure(
        [
            go.Bar(
                x=state_modes,
                y=np.mean(results, axis=1),
                error_y=dict(type="data", array=np.std(results, axis=1)),
            )
        ]
    )
    fig.write_html(f"data/n_collisions_plot.html")
    fig.show()
