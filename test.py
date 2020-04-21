from experiment import Experiment
from utils import get_conf, GraphWindow
from logger import Logger


class Exp(Experiment):
    def run(self):
        for i in range(10000):
            input()
            self.env.step([0, 1, 0, -1, 0, 0, 0])
            self_collision = self.env.check_collision_with_self()
            other_collision = self.env.check_collision()
            dynamic_collision = self.env.check_collision_with_dynamic()
            print(
                f"self collision: {self_collision}\nother collision: {other_collision}\ndynamic collision: {dynamic_collision}"
            )


cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = Exp(cnf, 0)
exp.run()
