from env.environment import Env
from conf import get_conf
from utils import ReturnWindow
from multiprocessing import Process, Array
import numpy as np

cnf = get_conf("conf/cnt_col.yaml")
env = Env(cnf)
state = env.reset()
done = False
timestep = 0
win = ReturnWindow(lookback=50)

# for i in range(32):
#     obs, *_, info = env.step([0, 0, 0, 0, 0, 1, -1])

"""
a = np.array([[2351, 236, 1090, 126, 8912], [6222, 7365,  120,  948, 5490], [5123, 3799,  3747, 1549, 1023]],)

"""


def run(rank, mode, cnf, results):
    np.random.seed()
    print("rank", rank, "writing results.")
    results[rank] = np.random.randint(99)


if __name__ == "__main__":
    processes = []
    results = Array('d', range(4))
    for rank in range(4):
        p = Process(target=run, args=(rank, "notrain", cnf, results))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print(results[:])
