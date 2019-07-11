import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score
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

    def inverse_transform(self, data):
        masked = ma.masked_invalid(data)
        masked *= self.stds
        masked += self.means
        return ma.getdata(masked)

class LinearLayer(nn.Module):
    def __init__(self, size_in, size_out, p=0.5):
        super(LinearLayer, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.p = p
        self.fc = nn.Linear(self.size_in, self.size_out)
        self.dropout = nn.Dropout(self.p)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(size_out)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)
    def forward(self, x):
        return self.dropout(self.relu(self.batchnorm(self.fc(x))))
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = LinearLayer(1825, 512, 0.5)
        self.linear2 = LinearLayer(512, 256, 0.5)
        self.linear3 = LinearLayer(256, 128, 0.5)
        self.linear4 = LinearLayer(128, 64, 0.5)
        self.linear5 = LinearLayer(64, 32, 0.5)
        self.linear6 = LinearLayer(32, 29, 0.1)
        self.linear7 = LinearLayer(29, 29, 0.0)
        self.layers = [self.linear1,self.linear2, self.linear3, self.linear4, self.linear5,self.linear6,self.linear7]
    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output
def mse(y_true, y_pred):
        return np.mean((y_true- y_pred)**2)

class ToxicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mask = ~ma.masked_invalid(self.y).mask
        self.y = np.nan_to_num(self.y)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (torch.from_numpy(self.x[idx]),torch.from_numpy(self.y[idx]),torch.from_numpy(self.mask[idx]))
if __name__ == "__main__":
    input_scaler = OurRobustToNanScaler()
    output_scaler = OurRobustToNanScaler()
    #with open("input_scaler.pk") as f:
    #    input_scaler = pickle.load(f)
    #
    #with open("output_scaler.pk") as f:
    #    input_scaler = pickle.load(f)

    x = np.load("morded112.npy")
    df = pd.read_csv("db_endpoints.csv")
    smiles = list(df[df.columns[0]])
    del df[df.columns[0]]
    y = df.values
    x = input_scaler.fit_transform(x)
    y = output_scaler.fit_transform(y)


    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    train_dataset = ToxicDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    test_dataset = ToxicDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    device = torch.device("cuda")
    model = MLP()
    model.load_state_dict(torch.load('mlp_epoch5000.pt'))
    model.to(device)

    model.eval()
    x_s = []
    y_s = []
    masks = []
    outputs =[]
    for batch_idx, (x, y, mask) in enumerate(test_loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        x = x.float()
        y = y.float()
        mask = mask.float()
        x_s.append(x.detach().cpu().numpy())
        y_s.append(y.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())
        output = model(x)
        outputs.append(output.detach().cpu().numpy())

    x_s = np.vstack(x_s)
    y_s = np.vstack(y_s)
    masks = np.vstack(masks)
    outputs = np.vstack(outputs)
    masks = np.array(masks, dtype= bool)

    y_s[~masks] = np.nan
    outputs[~masks]= np.nan
    y_s = output_scaler.inverse_transform(y_s)
    output_scaler = output_scaler.inverse_transform(outputs)

    
    mse_for_diff_endpoints = []
    for i in range(y_s.shape[1]):
        mse_for_diff_endpoints.append(mse((masks*y_s)[:, i][np.array(masks[:,i], dtype = bool)],(masks*outputs)[:, i][np.array(masks[:,i], dtype = bool)]) )


    r2score_for_diff_endpoints = []
    for i in range(y_s.shape[1]):
        r2score_for_diff_endpoints.append(r2_score((masks*y_s)[:, i][np.array(masks[:,i], dtype = bool)],(masks*outputs)[:, i][np.array(masks[:,i], dtype = bool)]) )

    with open("mse_r2.txt", "w") as f:
        for type, mse, r2_score_val in zip (list(df.columns), mse_for_diff_endpoints, r2score_for_diff_endpoints):
            f.write(type+" "+ str(mse)+" "+str(r2_score_val)+"\n")
