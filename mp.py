from multiprocessing import Array, Process


class Runner:
    def __init__(self, exp, cnf, name):
        self.exp = exp
        self.cnf = cnf
        self.name = name
    
    def run(self, n_procs):
        processes = []
        for rank in range(n_procs):
            print("startinc proc")
            p = Process(target=self._start_env, args=([], rank))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join()
    
    def _start_env(self, callbacks, rank):
        env = self.exp(cnf=self.cnf, name=self.name, rank=rank)
        env.run(callbacks)
        