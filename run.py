import os
import sys
import importlib
import multiprocessing as mp
from omegaconf import OmegaConf
from mp_runner import Runner


def get_conf(path):
    cnf = OmegaConf.load(path)
    cnf.merge_with_cli()
    OmegaConf.set_struct(cnf, True)
    return cnf


if __name__ == "__main__":

    mp.set_start_method("spawn")

    # dynamicall import the experiment
    experiment = sys.argv[1].split("/")[-1].split(".")[0]
    module = importlib.import_module("." + experiment, package="experiments")
    exp = getattr(module, "Experiment")

    cnf = get_conf("conf/main.yaml")
    if not cnf.wandb.name:
        os.environ["WANDB_MODE"] = "dryrun"
        os.environ["WANDB_DISABLE_CODE"] = "true"
    runner = Runner(exp, cnf)
    runner.run()
