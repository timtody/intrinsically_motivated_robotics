from multiprocessing import Process, Array


class Runner:
    def __init__(self, exp, cnf):
        self.exp = exp
        self.cnf = cnf

    def run(self, n_procs):
        results = Array('d', range(n_procs))
        processes = []
        for rank in range(n_procs):
            print("startinc proc")
            p = Process(target=self._start_env, args=([], rank, results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def _start_env(self, callbacks, rank, results):
        env = self.exp(self.cnf, rank)
        results[rank] = env.run(callbacks)
