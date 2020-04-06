from logger import Logger
from conf import get_conf
from experiment import CountCollisions

import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import multiprocessing as mp
from multiprocessing import Array, Process
from matplotlib import pyplot as plt


sns.set()


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
    results = {}
    state_modes = ["audio", "all"]

    for mode in state_modes:
        results[mode] = run_mode_mp(mode, cnf)

    results_df = pd.DataFrame(data=results)

    results = np.array(results)
    with open(f"data/n_collisions.p", "wb") as f:
        pickle.dump(results, f)

    results_df.mean(axis=0).plot.bar(yerr=results_df.std(axis=0)).set_title(
        "n collisions"
    )
    plt.savefig("data/n_collisions.png")
    plt.show()
