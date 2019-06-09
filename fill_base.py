import pandas as pd
import numpy as np
from rdkit import Chem
import rdkit.Chem.inchi as inchi
import random
import string

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

def fill_base(cursor):
    df = pd.read_csv("aggregate_tox.csv")
    df = df.drop("Unnamed: 0", axis=1)
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