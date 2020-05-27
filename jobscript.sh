#!/bin/bash
#SBATCH --job-name=foo
#SBATCH --partition=fuchs
#SBATCH --nodes=1
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000
#SBATCH --time=21-0
#SBATCH --no-requeue
#SBATCH --mail-type=FAIL
#SBATCH –-extra-node-info=2:10:1

python run.py experiments/count_collisions.py mp.n_procs=10 wandb.dryrun=False wandb.project=collisions-new wandb.name=test-sbatch &
python run.py experiments/count_collisions.py mp.n_procs=10 wandb.dryrun=False wandb.project=collisions-new wandb.name=test-sbatch &
wait