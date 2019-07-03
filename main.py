from create_database import create_database
from create_database import open_database
from fill_base import fill_base_test
from test_database import test_base
from fill_base import fill_base_real
import mpi4py
from mpi4py import MPI
if __name__ == "__main__":
    db_name = "chemdatabase.db"
    #create_database(db_name)
    conn, cursor = open_database(db_name)

    #comm = MPI.COMM_WORLD

    #print("Hello! I'm rank %d from %d running in total..." % (comm.rank, comm.size))


    #comm.Barrier()  # wait for everybody to synchronize _here_
    #fill_base_test(cursor)
    #test_base(cursor)
    fill_base_real(cursor,conn, name_of_file="melting100.csv", name_of_task="TestTask1000", name_of_descriptor="mordred", version_of_descriptor="1.0")

    conn.commit()
    cursor.close()
    conn.close()
