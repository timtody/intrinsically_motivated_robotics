
from mpi4py import MPI
nproc = MPI.COMM_WORLD.Get_size()
print(nproc)
