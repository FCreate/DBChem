from create_database import create_database
from create_database import open_database
from fill_base import fill_base_test
from test_database import test_base
import mpi4py
from mpi4py import MPI
if __name__ == "__main__":
    db_name = "chemdatabase.db"
    create_database(db_name)
    conn, cursor = open_database(db_name)

    #comm = MPI.COMM_WORLD

    #print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))


    #comm.Barrier()  # wait for everybody to synchronize _here_
    fill_base_test(cursor)
    test_base(cursor)
    conn.commit()
    cursor.close()
    conn.close()
