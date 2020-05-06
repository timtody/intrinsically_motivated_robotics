import os

from experiment import TestFWModel, CreateFWDB, GoalBasedExp, MeasureForgetting
import multiprocessing as mp
from runner import Runner
from utils import get_conf

if __name__ == "__main__":

    # os.environ["WANDB_MODE"] = "dryrun"
    # os.environ["WANDB_DISABLE_CODE"] = "true"

    mp.set_start_method("spawn")

    cnf = get_conf("conf/main.yaml")
    runner = Runner(MeasureForgetting, cnf)
    runner.run()
