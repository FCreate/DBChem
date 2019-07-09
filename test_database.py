from fill_base import morded
import pandas as pd
import numpy as np
def test_element(cursor, smile, endpoints,NameOfEndpoint2db_index, function):
    cursor = cursor.execute("select * from molecules where canonical_smiles='"+smile+"'")
    molecule_id = list(cursor)[0][0]
    
    experimental_data = cursor.execute("select * from experimental_data where id_molecule="+str(molecule_id)).fetchall()
    
    endpoints_from_table = list(endpoints)
    endpoints_indexes = list(endpoints.index)
    temp_data = [(molecule_id,NameOfEndpoint2db_index[endpoints_indexes[i+1]],endpoints_from_table[i+1]) for i in range(len(endpoints_from_table[1:]))]
    endpoints_flag = all(i in experimental_data for i in temp_data)
    
    temp = list(cursor.execute("select * from descriptors_values where id_molecule="+str(molecule_id)))
    #This need be fix. Working only for modrded.
    temp = temp[-1]
    descriptors_flag = (np.nan_to_num(np.array(morded(smile))) == np.nan_to_num(np.array(temp[4]))).all()
    if (endpoints_flag and descriptors_flag):
        return True
    else:
        return False
    
def test_base(cursor, csv_file, number_of_random_choices=10):
    df = pd.read_csv(csv_file)
    smiles = list(df[df.columns[0]])
    
    temp =list(cursor.execute("select * from endpoints").fetchall())
    temp = [(data[0], data[1]+"_"+data[2]) for data in temp]
    NameOfEndpoint2db_index = {data[1]:data[0] for data in temp}
    
    #test first element
    test_arr = []
    molecule_id_source = 0
    test_arr.append(test_element(cursor, smiles[molecule_id_source], df.loc[molecule_id_source].dropna(), NameOfEndpoint2db_index, morded))
    
    molecule_id_source = len(smiles)-1
    test_arr.append(test_element(cursor, smiles[molecule_id_source], df.loc[molecule_id_source].dropna(), NameOfEndpoint2db_index, morded))
    
    indices = np.random.randint(0, len(smiles), size = number_of_random_choices)
    for index in indices:
        test_arr.append(test_element(cursor, smiles[index], df.loc[index].dropna(), NameOfEndpoint2db_index, morded))
    return all(test_arr)

    
    
    
    
    
    