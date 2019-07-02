import pandas as pd
import numpy as np
from rdkit import Chem
import rdkit.Chem.inchi as inchi
import random
import string
from mpi4py import MPI
import rdkit
from mordred import Calculator, descriptors
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

def canonize_smile (sm):
    m = Chem.MolFromSmiles(sm)
    try: return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)
    except: return None

def _canonize_mixture (mix):
    return '.'.join([canonize_smile(sm) for sm in mix.split('.')])

def randomStringwithDigitsAndSymbols(stringLength=10):
    """Generate a random string of letters, digits and special characters """
    password_characters = string.ascii_letters + string.digits
    return ''.join(random.choice(password_characters) for i in range(stringLength))
def func(smiles):
    value = [np.random.randn(random.randint(1,1825),1) for i in range(len(smiles))]
    return value



morded_calculator = Calculator(descriptors, ignore_3D=False)

fprint_params = {'bits': 4096, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
confgen_params = {'max_energy_diff': 20.0, 'first': 10}

def ecfp( mol, r=3, nBits=4096, errors_as_zeros=True):
    mol = Chem.MolFromSmiles(mol) if not isinstance(mol, rdkit.Chem.rdchem.Mol) else mol
    try:
        arr = np.zeros((1,))
        ConvertToNumpyArray(GetMorganFingerprintAsBitVect(mol, r, nBits), arr)
        return arr.astype(np.float32)
    except:
        return np.NaN if not errors_as_zeros else np.zeros((nBits,), dtype=np.float32)

def ecfp_for_builder(x): return np.array([ecfp(x)]).astype(np.float32)
#def e3fp(smiles): return np.vstack([fp.to_vector(sparse=False).astype(np.float32) for fp in fprints_from_smiles(smiles,smiles,confgen_params,fprint_params)])
def morded(smiles): return np.array([list(morded_calculator(Chem.MolFromSmiles(smiles))
                                      .fill_missing(value=0.)
                                      .values())])\
    .astype(np.float32)

def get_avaliable_descriptors(): return {'morded':morded,'ecfp':ecfp}


def fill_base_test(cursor):
    df = pd.read_csv("toxicity_85832.csv")
    #df = df.drop("Unnamed: 0", axis=1)
    names_of_columns = list(df.columns)
    smiles= list(df["SMILES"])
    df = df.drop("SMILES", axis = 1)
    toxic_vals = np.array(df.values)

    #molecules
    canonize_smiles = [_canonize_mixture(smile) for smile in smiles]
    inchi_smiles = [inchi.MolToInchi(Chem.MolFromSmiles(smile))for smile in canonize_smiles]
    inchikey = [inchi.MolToInchiKey(Chem.MolFromSmiles(smile)) for smile in canonize_smiles]
    ids = [x for x in range(len(canonize_smiles))]
    ziped_vals = zip(inchikey, inchi_smiles, canonize_smiles)
    cursor.executemany("""insert into 'molecules' (inchi_key,inchi,canonical_smiles) values (?,?,?)""", ziped_vals)

    #tasks
    descr_tasks = [randomStringwithDigitsAndSymbols(random.randint(1,30)) for i in range(20)]
    cursor.executemany("""insert into 'tasks' (descr) values (?)""", zip(descr_tasks))

    #tasks_running
    completed = [random.randint(0,1) for i in range (1000)]
    id_tasks= [random.randint(1,len(descr_tasks))for i in range(1000)]
    id_molecules = [random.randint(1,len(smiles))for i in range(1000)]
    zip_tasks_running = zip(id_tasks, id_molecules, completed)
    cursor.executemany("""insert into 'tasks_running' (id_task, id_molecule, completed) values (?,?,?)""", zip_tasks_running)

    #descriptors
    name_of_descr = [randomStringwithDigitsAndSymbols(random.randint(1,30)) for i in range(10)]
    name_of_version = [randomStringwithDigitsAndSymbols(random.randint(1,30)) for i in range(10)]
    ziped_versions = zip(name_of_descr, name_of_version)
    cursor.executemany("""insert into 'descriptors' (descriptor, version) values (?,?)""",  ziped_versions)
    cursor.execute("""insert into 'descriptors' (descriptor, version) values (?,?)""",("mordred","0.315"))

    #descriptor_values
    id_descriptor = [11 for i in range(len(smiles))]
    id_molecule = [x+1 for x in range(len(smiles))]
    id_tasks = [random.randint(1,len(descr_tasks)) for i in range(len(smiles))]
    valid = [random.randint(0,1) for i in range(len(smiles))]
    value = func( canonize_smiles)
    ziped_descr_vals = zip(id_molecule, id_descriptor, id_tasks, valid, value)
    cursor.executemany("""insert into 'descriptors_values' (id_molecule, id_descriptor, id_task, valid, value) values (?,?,?,?,?)""",  ziped_descr_vals)

    #endpoints
    features = names_of_columns[1:]
    descriptions = [feature.split('_')[1] for feature in features]
    types = ['_'.join(feature.split('_')[2:]) for feature in features]
    ziped_endpoints = zip(descriptions, types)
    cursor.executemany("""insert into 'endpoints' (desc, type) values (?,?)""",  ziped_endpoints)

    #experimnetal data
    ids_molecules = []
    ids_endpoints = []
    values_endpoints = []
    for i in range(len(toxic_vals[:,0])):
        for j in range(len(toxic_vals[0,:])):
            if (~np.isnan(toxic_vals[i,j])):
                ids_molecules.append(i+1)
                ids_endpoints.append(j+1)
                values_endpoints.append(toxic_vals[i,j])

    ziped_experimental_data = zip(ids_molecules, ids_endpoints, values_endpoints)
    cursor.executemany("""insert into 'experimental_data' (id_molecule, id_endpoint, value) values (?,?,?)""",  ziped_experimental_data)

    return cursor
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
def fill_base_real(cursor, conn, name_of_task, descr_task):
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    if (comm.rank==0):
        name_of_task = "Hello world"
        cursor.execute("""insert into 'tasks' (descr) values (?)""", (name_of_task,))
        conn.commit()

        id_task = list(cursor.execute("SELECT last_insert_rowid()"))[0][0]

        df_melt = pd.read_csv("melting_prepared_282517.csv")
        names_of_columns = list(df_melt.columns)
        smiles = list(df_melt[df_melt.columns[0]])
        #smiles = [canonize_smile(smile) for smile in smiles]

        smiles_tuple = "("
        for smile in smiles:
            smiles_tuple += "'" + smile + "', "
        smiles_tuple = smiles_tuple[:-2]
        smiles_tuple += ")"

        existed_smiles = list(cursor.execute("""Select * From molecules Where canonical_smiles in """ + smiles_tuple))
        only_existed_smiles = [data[3] for data in existed_smiles]

        not_existed_smiles = (list(set(smiles) - set(only_existed_smiles)))
        print(len(list(set(only_existed_smiles))) + len(list(set(not_existed_smiles))) == len(set(smiles)))
        print(len(not_existed_smiles))
        # Вот эту часть нужно распараллелить
        size_per_processor = round(len(not_existed_smiles)/size)+1
        chunked_smiles = list(chunks(not_existed_smiles, size_per_processor))
        my_smiles = chunked_smiles[0]
        for i in range(1, size):
            comm.send(chunked_smiles[i], dest= i, tag=0)

        my_inchi_smiles = [inchi.MolToInchi(Chem.MolFromSmiles(smile)) for smile in my_smiles]
        my_inchikey = [inchi.MolToInchiKey(Chem.MolFromSmiles(smile)) for smile in my_smiles]

        inchi_smiles = []
        inchikey = []
        inchi_smiles.extend(my_inchi_smiles)
        inchikey.extend(my_inchikey)
        for i in range(1, size):
            my_inchi_smiles = comm.recv(source = i,tag=0)
            my_inchikey= comm.recv(source= i, tag=0)
            inchi_smiles.extend(my_inchi_smiles)
            inchikey.extend(my_inchikey)

        ziped_vals = zip(inchikey, inchi_smiles, smiles)
        cursor.executemany("""insert into 'molecules' (inchi_key,inchi,canonical_smiles) values (?,?,?)""", ziped_vals)
        conn.commit()

        existed_smiles = list(cursor.execute("""Select * From molecules Where canonical_smiles in """ + smiles_tuple))
        smiles2ind = {}
        for data in existed_smiles:
            smiles2ind[data[3]] = data[0]

        names_of_columns = list(df_melt.columns)
        features = names_of_columns[1:]
        descriptions = [feature.split('_')[0] for feature in features]
        types = ['_'.join(feature.split('_')[1:]) for feature in features]
        ziped_endpoints = zip(descriptions, types)
        cursor.executemany("""insert into 'endpoints' (desc, type) values (?,?)""", ziped_endpoints)
        conn.commit()

        endpoint2ind = {}
        db_endp = list(cursor.execute("""SELECT * FROM endpoints"""))
        db_endp = [(data[0], data[1] + "_" + data[2]) for data in db_endp]
        for name in db_endp:
            endpoint2ind[name[1]] = name[0]

        df_melt = df_melt.drop(df_melt.columns[0], axis=1)
        endpoints_vals = df_melt.values

        # experimnetal data
        ids_molecules = []
        ids_endpoints = []
        values_endpoints = []
        for i in range(len(endpoints_vals[:, 0])):
            for j in range(len(endpoints_vals[0, :])):
                if (~np.isnan(endpoints_vals[i, j])):
                    ids_molecules.append(smiles2ind[smiles[i]])
                    ids_endpoints.append(endpoint2ind[names_of_columns[j + 1]])
                    values_endpoints.append(endpoints_vals[i, j])

        ziped_experimental_data = zip(ids_molecules, ids_endpoints, values_endpoints)
        cursor.executemany("""insert into 'experimental_data' (id_molecule, id_endpoint, value) values (?,?,?)""",
                           ziped_experimental_data)
        conn.commit()

        #Начинаем считать дескрипторы
        my_mordred_descriptors = []
        for smile in my_smiles:
            temp = morded(smile)
            if (not temp is None):
                my_mordred_descriptors.append(np.array(morded(smile)).flatten())
            else:
                my_mordred_descriptors.append(np.array([0]*1826))
        mordred_final = my_mordred_descriptors.copy()
        for i in range(1,size):
            temp_mordred = np.empty(len(chunked_smiles[i])*1826, dtype=np.float32)
            comm.Recv(temp_mordred, source=i, tag=1)
            temp_mordred = temp_mordred.reshape(-1, 1826)
            mordred_final = np.concatenate((mordred_final, temp_mordred), axis = 0)
        print(mordred_final.shape)


    if(rank!=0):
        my_smiles = comm.recv(source = 0, tag=0)

        my_inchi_smiles = [inchi.MolToInchi(Chem.MolFromSmiles(smile)) for smile in my_smiles]
        my_inchikey = [inchi.MolToInchiKey(Chem.MolFromSmiles(smile)) for smile in my_smiles]

        comm.send(my_inchi_smiles, dest = 0, tag = 0)
        comm.send(my_inchikey, dest = 0, tag = 0)

        my_mordred = []
        for smile in my_smiles:
            temp = morded(smile)
            if (not temp is None):
                my_mordred.append(np.array(temp).reshape(1826))
            else:
                my_mordred.append(np.array([0] * 1826))
        my_mordred = np.vstack(my_mordred).flatten()
        print("i'm processor number ", rank, flush=True)
        comm.Send(my_mordred, dest=0, tag=1)

