import sqlite3
import numpy as np
import io


def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


def create_database(db_name):
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)

    conn = sqlite3.connect(db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""CREATE TABLE IF NOT EXISTS molecules
                      (id_molecule integer PRIMARY KEY AUTOINCREMENT ,
                       inchi_key text NOT NULL,
                       inchi text NOT NULL,
                       canonical_smiles text NOT NULL);
                      """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS tasks  
                      (id_task integer PRIMARY KEY AUTOINCREMENT ,
                       descr text NOT NULL)
                      """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS tasks_running  
                      (id_task integer, 
                       id_molecule integer,
                       completed integer,
                       FOREIGN KEY (id_task) REFERENCES tasks(id_task) ON UPDATE NO ACTION  ON DELETE NO ACTION,
                       FOREIGN KEY (id_molecule) REFERENCES molecules(id_molecule) ON UPDATE  NO ACTION ON DELETE NO ACTION );
                      """)
    cursor.execute("""CREATE TABLE IF NOT EXISTS descriptors  
                      (id_descriptor integer PRIMARY KEY AUTOINCREMENT ,
                       descriptor text, 
                       version text);

                      """)
    cursor.execute("""CREATE TABLE IF NOT EXISTS descriptors_values 
                      (id_molecule integer,
                       id_descriptor integer,
                       id_task integer, 
                       valid integer,
                       value array,
                       FOREIGN KEY (id_task) REFERENCES tasks(id_task) ON UPDATE NO ACTION  ON DELETE NO ACTION,
                       FOREIGN KEY (id_molecule) REFERENCES molecules(id_molecule) ON UPDATE  NO ACTION ON DELETE NO ACTION,
                       FOREIGN KEY (id_descriptor) REFERENCES descriptors(id_descriptor) ON UPDATE  NO ACTION ON DELETE NO ACTION);
                      """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS endpoints  
                      (id_endpoint integer PRIMARY KEY AUTOINCREMENT,
                       desc text,
                       type text);
                      """)

    cursor.execute("""CREATE TABLE IF NOT EXISTS experimental_data
                      (id_molecule integer,
                       id_endpoint integer,
                       value float,
                       FOREIGN KEY (id_molecule) REFERENCES molecules(id_molecule) ON UPDATE  NO ACTION ON DELETE NO ACTION,
                       FOREIGN KEY (id_endpoint) REFERENCES endpoints(id_endpoint) ON UPDATE  NO ACTION ON DELETE NO ACTION);
                      """)
    conn.commit()
    conn.close()

def open_database(db_name):
    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)
    conn = sqlite3.connect(db_name, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    return conn, cursor
