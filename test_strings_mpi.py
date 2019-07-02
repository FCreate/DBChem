from mpi4py import MPI
import pandas as pd
import numpy as np
from rdkit import Chem
import rdkit.Chem.inchi as inchi

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
import rdkit
from mordred import Calculator, descriptors
from rdkit import Chem
import numpy as np
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

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
if __name__ == "__main__":
    # comm =  MPI.COMM_WORLD
    # rank = comm.rank
    # size = comm.size
    # if (rank == 0):
    #     l = ["Hello", "world"]
    #     comm.send(l, dest=1, tag = 1)
    #     print("I' m a processor with rank ", rank, "It's my data ", l)
    # if(rank ==1):
    #     l = comm.recv(source = 0, tag = 1)
    #     print("I' m a processor with rank ", rank, "It's my data ", l)

    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    if (comm.rank == 0):

        df_melt = pd.read_csv("melting_prepared_282517.csv")
        names_of_columns = list(df_melt.columns)
        smiles = list(df_melt[df_melt.columns[0]])
        # smiles = [canonize_smile(smile) for smile in smiles]

        # Вот эту часть нужно распараллелить
        smiles = smiles[:1000]
        not_existed_smiles = smiles
        size_per_processor = round(len(not_existed_smiles) / size) + 1
        chunked_smiles = list(chunks(not_existed_smiles, size_per_processor))
        my_smiles = chunked_smiles[0]
        for i in range(1, size):
            comm.send(chunked_smiles[i], dest=i, tag=0)

        my_inchi_smiles = [inchi.MolToInchi(Chem.MolFromSmiles(smile)) for smile in my_smiles]
        my_inchikey = [inchi.MolToInchiKey(Chem.MolFromSmiles(smile)) for smile in my_smiles]

        inchi_smiles = []
        inchi_key = []
        inchi_smiles.extend(my_inchi_smiles)
        inchi_key.extend(my_inchikey)
        for i in range(1, size):
            my_inchi_smiles = comm.recv(source=i, tag=0)
            my_inchikey = comm.recv(source=i, tag=0)
            inchi_smiles.extend(my_inchi_smiles)
            inchi_key.extend(my_inchikey)
        print(len(inchi_smiles))
        print(len(inchi_key))

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
            comm.Recv(temp_mordred, dest=i, tag=0)
            temp_mordred = temp_mordred.reshape(-1, 1826)
            mordred_final = np.concatenate((mordred_final, temp_mordred), axis = 0)
        print(mordred_final.shape)


    if (rank != 0):
        my_smiles = comm.recv(source=0, tag=0)

        my_inchi_smiles = [inchi.MolToInchi(Chem.MolFromSmiles(smile)) for smile in my_smiles]
        my_inchikey = [inchi.MolToInchiKey(Chem.MolFromSmiles(smile)) for smile in my_smiles]

        comm.send(my_inchi_smiles, dest=0, tag=0)
        comm.send(my_inchikey, dest=0, tag=0)
        my_mordred = []
        for smile in my_smiles:
            temp = morded(smiles[i])
            if (not temp is None):
                my_mordred.append(np.array(temp).reshape(1826))
            else:
                my_mordred.append(np.array([0] * 1826))
        my_mordred = np.vstack(my_mordred).flatten()
        comm.Send(my_mordred, source = 0, tag = 0)