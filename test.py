from experiment import (
    TestFWModel,
    CountCollisionsAgent,
    MeasureForgetting,
    GoalBasedExp,
    Experiment,
    TestStateDifference,
    CreateFWDB,
    TestFWModelFromDB,
)

from logger import Logger
from utils import get_conf, GraphWindow

import time
import os
import numpy as np
import multiprocessing as mp


class Exp(Experiment):
    def run(self):
        obs = self.env.reset()
        print(len(obs))
        exit(1)
        for i in range(15):
            self.env.step([0, 0, 0, 0, 0, 1, 0])

        for i in range(200):
            self.env.step([0, 1, 0, 0, 0, 0, 0])

        self.env.close()


if __name__ == "__main__":

    np.set_printoptions(suppress=True)

    # os.environ["WANDB_MODE"] = "dryrun"
    # os.environ["WANDB_DISABLE_CODE"] = "true"

    mp.set_start_method("spawn")

    cnf = get_conf("conf/main.yaml")
    # TODO: remove this when loading forgetting db
    exp = GoalBasedExp(cnf, 0)
    exp.run()
