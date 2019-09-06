import numpy.ma as ma
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y))
        return loss
class LinearLayer(nn.Module):
    def __init__(self, size_in, size_out, p=0.5, is_bn =True, activation = True):
        super(LinearLayer, self).__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.is_bn = is_bn
        self.activation = activation
        self.p = p
        self.fc = nn.Linear(self.size_in, self.size_out)
        self.dropout = nn.Dropout(self.p)
        self.elu = nn.ELU()
        if self.is_bn:
            self.batchnorm = nn.BatchNorm1d(size_out)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.fill_(0.01)
    def forward(self, x):
        output = self.fc(x)
        if self.is_bn:
            output = self.batchnorm(output)
        if self.activation:
            output = self.elu(output)
        output = self.dropout(output)
        return output
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = LinearLayer(1826, 512, 0.5)
        # self.linear1_5 = LinearLayer(4096, 2048, 0.5)
        # self.linear1_7 = LinearLayer(2048, 1024, 0.5)
        # self.linear1_8 = LinearLayer(4096, 512, 0.5)
        # self.linear1_5 = LinearLayer(1024, 512, 0.5)
        self.linear2 = LinearLayer(512, 256, 0.5)
        self.linear3 = LinearLayer(256, 128, 0.5)
        self.linear4 = LinearLayer(128, 64, 0.5)
        self.linear5 = LinearLayer(64, 32, 0.25)
        self.linear6 = LinearLayer(32, 29, 0.1)
        self.linear7 = LinearLayer(29, 29, 0.0, is_bn=False, activation=False)
        self.layers = [self.linear1, self.linear2, self.linear3, self.linear4, self.linear5,self.linear6,self.linear7]
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
        return (torch.from_numpy(self.x[idx]),torch.from_numpy(np.float32(self.y[idx])),torch.from_numpy(np.float32(self.mask[idx])))

def run_model(X_train, X_test, y_train, y_test, cuda, log_file_name, log_dir_name, number_of_epochs, ckpt_name, scaler):
    if cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = MLP().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    if log_file_name is not None:
        log_file = open(log_file_name, 'w')

    if log_dir_name is not None:
        writer = writer = SummaryWriter(log_dir=log_dir_name)

    train_dataset = ToxicDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True)

    test_dataset = ToxicDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

    train_loss = []
    test_loss = []

    def train(epoch):
        model.train()
        losses = []
        for batch_idx, (x, y, mask) in enumerate(train_loader):
            x, y, mask = x.to(device, ), y.to(device), mask.to(device)
            optimizer.zero_grad()
            output = model(x)
            output.squeeze_(dim=1)

            loss_func = RMSELoss()
            loss = loss_func(mask * output, mask * y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        return sum(losses) / len(losses)

    def test(epoch):
        model.eval()
        losses = []
        for batch_idx, (x, y, mask) in enumerate(test_loader):
            x, y, mask = x.to(device), y.to(device), mask.to(device)

            output = model(x)
            output.squeeze_(dim=1)

            loss_func = RMSELoss()
            loss = loss_func(mask * output, mask * y)
            losses.append(loss.item())
        return sum(losses) / len(losses)

    for epoch in range(1, number_of_epochs + 1):
        loss = train(epoch)
        train_loss.append(loss)
        if log_file_name is not None:
            log_file.write("Epoch{} ".format(epoch))
            log_file.write(str(loss))
            log_file.write("\n")
        if log_dir_name is not None:
            writer.add_scalar('train', loss, epoch)

        loss = test(epoch)
        if (((len(test_loss)) == 0) or (loss < min(test_loss))):
            torch.save(model.state_dict(), ckpt_name)
        if log_file_name is not None:
            log_file.write("Epoch{} ".format(epoch))
            log_file.write(str(loss))
            log_file.write("\n")
            log_file.flush()
        if log_dir_name is not None:
            writer.add_scalar('test', loss, epoch)
        test_loss.append(loss)

    model.load_state_dict(torch.load(ckpt_name))
    model.to(device)

    model.eval()
    x_s = []
    y_s = []
    masks = []
    outputs = []
    for batch_idx, (x, y, mask) in enumerate(test_loader):
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        x_s.append(x.detach().cpu().numpy())
        y_s.append(y.detach().cpu().numpy())
        masks.append(mask.detach().cpu().numpy())
        output = model(x)
        outputs.append(output.detach().cpu().numpy())

    x_s = np.vstack(x_s)
    y_s = np.vstack(y_s)
    masks = np.vstack(masks)
    outputs = np.vstack(outputs)

    outputs = scaler.inverse_transform(outputs)
    y_s = scaler.inverse_transform(y_s)

    mse_for_diff_endpoints = []
    for i in range(y_s.shape[1]):
        mse_for_diff_endpoints.append(mse((masks * y_s)[:, i][np.array(masks[:, i], dtype=bool)],
                                          (masks * outputs)[:, i][np.array(masks[:, i], dtype=bool)]))

    r2score_for_diff_endpoints = []
    for i in range(y_s.shape[1]):
        r2score_for_diff_endpoints.append(r2_score((masks * y_s)[:, i][np.array(masks[:, i], dtype=bool)],
                                                   (masks * outputs)[:, i][np.array(masks[:, i], dtype=bool)]))

    return mse_for_diff_endpoints, r2score_for_diff_endpoints