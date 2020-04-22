from experiment import Experiment
from utils import get_conf, GraphWindow
from logger import Logger
import time


class Exp(Experiment):
    def run(self):
        win = GraphWindow(
            [
                "mobile_0_alpha",
                "mobile_0_gamma",
                "mobile_1_alpha",
                "mobile_1_gamma",
                "mobile_2_alpha",
                "mobile_2_gamma",
            ],
            2,
            3,
        )

        start = time.time()
        for i in range(1000):
            # self.env._mobile_2_joint_0.set_joint_target_velocity(0)
            # self.env._mobile_2_joint_1.set_joint_target_velocity(0)
            # self.env._mobile_1_joint_0.set_joint_target_velocity(0)
            # self.env._mobile_1_joint_1.set_joint_target_velocity(0)
            # self.env._mobile_0_joint_0.set_joint_target_velocity(0)
            # self.env._mobile_0_joint_1.set_joint_target_velocity(0)
            obs, *_ = self.env.step([0, 1, 0, -1, 0, 0, 0])
            print(obs)
            print(len(obs))
            exit(1)
            self_collision = self.env.check_collision_with_self()
            other_collision = self.env.check_collision()
            dynamic_collision = self.env.check_collision_with_dynamic()

            # win.update(*self.env.get_mobile_velocities())
            # print(
            #     f"self collision: {self_collision}\nother collision: {other_collision}\ndynamic collision: {dynamic_collision}"
            # )
        end = time.time()
        print("took", end - start, "s")


cnf = get_conf("conf/main.yaml")
Logger(cnf)
exp = Exp(cnf, 0)
exp.run()
