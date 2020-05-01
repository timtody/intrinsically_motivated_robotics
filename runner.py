import multiprocessing as mp


class Runner:
    def __init__(self, exp, cnf):
        self.exp = exp
        self.cnf = cnf

    def run(self):
        processes = []
        for rank in range(self.cnf.mp.n_procs):
            p = mp.Process(target=self._start_env, args=(rank,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    def _start_env(self, rank):
        self.cnf.env.torch_seed += rank
        self.cnf.env.np_seed += rank
        exp = self.exp(self.cnf, rank)
        exp.run()
