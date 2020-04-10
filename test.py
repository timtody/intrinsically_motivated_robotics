from experiment import Experiment
from utils import get_conf
from logger import Logger


class Exp(Experiment):
    def run(self):
        for i in range(10000):
            self.env.step(self.env.action_space.sample())


cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = Exp(cnf, 0)
exp.run()
