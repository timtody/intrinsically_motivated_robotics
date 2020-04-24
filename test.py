from experiment import Experiment
from utils import get_conf, GraphWindow
from logger import Logger
import time


class Exp(Experiment):
    def run(self):
        win = GraphWindow(["skin x", "skin y", "skin z",], 3, 1, autoscale=True,)

        start = time.time()
        for i in range(1000):
            # input()
            obs, *_ = self.env.step([0, 0.1, 0, -0.5, 0, -0.5, 0])
            # obs, *_ = self.env.step(self.env.action_space.sample())
            self_collision = self.env.check_collision_with_self()
            other_collision = self.env.check_collision()
            dynamic_collision = self.env.check_collision_with_dynamic()

            win.update(*self.env.skin_sensor_test.read()[0])
        end = time.time()
        print("took", end - start, "s")


cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = Exp(cnf, 0)
exp.run()
