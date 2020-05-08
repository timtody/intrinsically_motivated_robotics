import os
import sys
import ray
import importlib
import multiprocessing as mp
from exp_runner import Runner
from utils import get_conf

if __name__ == "__main__":

    mp.set_start_method("spawn")

    # dynamicall import the experiment
    experiment = sys.argv[1].split("/")[1].split(".")[0]
    module = importlib.import_module("." + experiment, package="experiments")
    exp = getattr(module, "Experiment")

    cnf = get_conf("conf/main.yaml")
    if cnf.wandb.dryrun:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["WANDB_DISABLE_CODE"] = "true"
    runner = Runner(exp, cnf)
    runner.run()
