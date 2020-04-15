from experiment import Experiment
from utils import get_conf, GraphWindow
from logger import Logger

win = GraphWindow(
    [
        "mobile_0_beta",
        "mobile_0_gamma",
        "mobile_1_beta",
        "mobile_1_gamma",
        "mobile_2_beta",
        "mobile_2_gamma",
    ],
    2,
    3,
)


class Exp(Experiment):
    def run(self):
        for i in range(10000):
            self.env.step(self.env.action_space.sample())
            win.update(*self.env.get_mobile_positions())


cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = Exp(cnf, 0)
exp.run()
