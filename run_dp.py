import os
import sys
import importlib
import multiprocessing as mp
from exp_runner import Runner
from utils import get_conf

from mpi4py import MPI


nproc = MPI.COMM_WORLD.Get_size()  # Size of communicator
iproc = MPI.COMM_WORLD.Get_rank()  # Ranks in communicator
inode = MPI.Get_processor_name()  # Node where this MPI process runs


# dynamically import the experiment
experiment = sys.argv[1].split("/")[1].split(".")[0]
module = importlib.import_module("." + experiment, package="experiments")
exp = getattr(module, "Experiment")

cnf = get_conf("conf/main.yaml")

if cnf.wandb.dryrun:
    os.environ["WANDB_MODE"] = "dryrun"
    os.environ["WANDB_DISABLE_CODE"] = "true"

experiment = exp(cnf, iproc)
experiment.run()

MPI.Finalize()
