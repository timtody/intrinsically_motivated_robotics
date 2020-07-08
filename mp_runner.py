import multiprocessing as mp
from multiprocessing import Array


class Runner:
    def __init__(self, exp, cnf):
        self.exp = exp
        self.cnf = cnf
        self.results = []

    def run(self):
        processes = []
        manager = mp.Manager()
        d = manager.dict()
        print('executing pre run hook')
        pre_run_results = self.exp.pre_run_hook()
        for rank in range(self.cnf.mp.n_procs):
            p = mp.Process(target=self._start_env,
                           args=(rank, d, pre_run_results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        self.exp.plot(d)

    def _start_env(self, rank, d, pre_run_results):
        self.cnf.env.torch_seed += rank
        self.cnf.env.np_seed += rank
        exp = self.exp(self.cnf, rank)
        results = exp.run(pre_run_results)
        d[rank] = results
