from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
if rank == 0:
    data = {'a': 7, 'b': 3.14}
    sent = [comm.isend(data, dest=r, tag=11) for r in range(comm.Get_size())]
    [s.wait() for s in sent]
else:
    data = comm.irecv(source=0, tag=11)
    print(str(rank) + " got " + str(data.wait()))