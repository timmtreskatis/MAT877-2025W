from mpi4py import MPI
import numpy as np

# Communicator = group of processes with communication interface
comm = MPI.COMM_WORLD  # world communicator = ALL processes in this run
# Rank = unique ID number of each process within this communicator
rank = comm.rank
# Size = total number of processes in this communicator
size = comm.size

print("Hello from rank", rank, "out of", size - 1)

# Scattering: one root process sends DIFFERENT PIECES OF DATA to all processes in the communicator
if rank == 0:
    data = [1, 2, 3, 4]
else:
    data = None

a = None

print("Rank", rank, "before scattering uses a =", a)

a = comm.scatter(data, root=0)

print("Rank", rank, "after scattering uses a =", a)

b = 2 * a


# Gathering = inverse of scattering: each process in the communicator sends its local data to one root process for collection
results = comm.gather(b, root=0)

print("Rank", rank, "after gathering uses results =", results)


# Broadcasting: one process sends THE SAME DATA to all others in the communicator
if rank == 0:
    average = np.mean(results)
else:
    average = None

print("Rank", rank, "before broadcasting knows that the average is", average)

average = comm.bcast(average, root=0)

print("Rank", rank, "after broadcasting knows that the average is", average)
