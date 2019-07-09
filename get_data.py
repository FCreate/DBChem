import yaml
from create_database import open_database
import pandas as pd
import numpy as np
import numpy.ma as ma

class OurRobustToNanScaler():
    """
    This class is equal to StandardScaler from sklearn but can work with NaN's (ignoring it) but
    sklearn's scaler can't do it.
    """

    def fit(self, data):
        masked = ma.masked_invalid(data)
        self.means = np.mean(masked, axis=0)
        self.stds = np.std(masked, axis=0)

    def fit_transform(self, data):
        self.fit(data)
        masked = ma.masked_invalid(data)
        masked -= self.means
        masked /= self.stds
        return ma.getdata(masked)

    def transform(self, data):
        masked = ma.masked_invalid(data)
        masked -= self.means
        masked /= self.stds
        return ma.getdata(masked)

    def inverse_transform(self, data):
        masked = ma.masked_invalid(data)
        masked *= self.stds
        masked += self.means
        return ma.getdata(masked)

def get_data(yaml_file):
    with open(yaml_file, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    endpoints_list = cfg["endpoints_list"]
    conn, cursor  = open_database(cfg["main"]["db_name"])

    #Create dictionary
    all_endpoints = cursor.execute("select * from endpoints").fetchall()
    all_endpoints = [(data[0], data[1].strip() + "_" + data[2].strip()) for data in all_endpoints]
    NameOfEndpoint2db_index = {data[1]: data[0] for data in all_endpoints}

    #create datafram
    indexes = [NameOfEndpoint2db_index[endpoint] for endpoint in cfg['endpoints_list']]
    temp = cursor.execute("select * from experimental_data where id_endpoint in " + ("(" + ", ".join(str(x) for x in indexes) + ")")).fetchall()
    endpoints_dict = {}

    for data in temp:
        if data[0] in endpoints_dict.keys():
            endpoints_dict[data[0]].append((data[1], data[2]))
        else:
            endpoints_dict[data[0]] = [(data[1], data[2])]
    indexe = [x + 1 for x in list(range(len(endpoints_list)))]

    for key in endpoints_dict.keys():
        vals = {idx: None for idx in indexe}
        for elem in endpoints_dict[key]:
            vals[elem[0]] = elem[1]
        endpoints_dict[key] = list(vals.values())

    db_smiles = cursor.execute("select * from molecules").fetchall()
    index2smiles = {smile[0]: smile[3] for smile in db_smiles}

    endpoints_keys = list(endpoints_dict.keys())
    for key in endpoints_keys:
        endpoints_dict[index2smiles[key]] = endpoints_dict.pop(key)
    #Scaling
    df = pd.DataFrame.from_dict(endpoints_dict, orient="index", columns=endpoints_list)
    endpoints_scaler = OurRobustToNanScaler()
    df[:][:] = endpoints_scaler.fit_transform(df.values)

    #get descriptors for molecules
    smiles_ind = list(set([data[0] for data in temp]))
    descriptors = cursor.execute("select * from descriptors_values where id_molecule in " + ("(" + ", ".join(str(x) for x in smiles_ind) + ")")).fetchall()
    ind2descriptors = {index2smiles[data[0]]:np.nan_to_num(data[4]) for data in descriptors}


