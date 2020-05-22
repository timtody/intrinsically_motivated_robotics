import os
import sys
import importlib
import multiprocessing as mp
from mp_runner import Runner
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

    modes = [
        "tac",
        "prop",
        "tac,prop",
    ]

    wandb_basename = cnf.wandb.name
    for mode in modes:
        print("starting mode", mode)
        cnf.env.state = mode
        cnf.wandb.name = wandb_basename + f"-{mode}"
        runner = Runner(exp, cnf)
        runner.run()
