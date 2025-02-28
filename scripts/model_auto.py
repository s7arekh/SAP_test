import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NARXAuto(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, M):
        self.M = M
        super(NARXAuto, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            # nn.Sigmoid(),
            # nn.Tanh(),
            # nn.ReLU(),
            *[
                layer
                for i in range(len(hidden_sizes) - 1)
                for layer in (nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]),)#, nn.Sigmoid())
            ],
            nn.Linear(hidden_sizes[-1], output_size),
            # nn.Softplus()
            # nn.ReLU()
        )

    def forward(self, x):
        out = self.net(x)
        return out
    
    def train_model(self, trainloader, test_data_X, test_data_y, steps=18, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_losses = []
        rmse_epochs = []

        for epoch in tqdm(range(epochs), desc='Training', unit='epoch'):
            self.train()
            train_loss = 0
            for X, y in trainloader:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()
                output = self(X)
                loss = criterion(output, y.unsqueeze(1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(trainloader))
            
            # Каждые 500 эпох считаем RMSE
            if (epoch + 1) % 10000 == 0:
                rmse = self.compute_rmse(test_data_X, test_data_y, steps)
                rmse_epochs.append(rmse)
        return np.array(train_losses), np.array(rmse_epochs)

    def autoregressive_prediction(self, data, start, steps):
        preds = []
        self.eval()
        X = data[start]
        for i in range(start, start + steps):
            with torch.no_grad():
                pred = self(X.to(device))
                preds.append(pred.item())
                model_s = data[i + 1, :5]
                obs_s = X[-3:]
                obs_s = torch.cat((obs_s, pred))
                X = torch.cat((model_s, obs_s))
        return preds

    def compute_rmse(self, test_data_X, test_data_y, steps):
        preds = []
        deltas = []
        for i in range(len(test_data_X) - steps):
            preds_auto = self.autoregressive_prediction(test_data_X.to(device), i, steps)
            target = test_data_y[i:i + steps].numpy()
            delta = (target - np.array(preds_auto)) * self.M
            preds.append(preds_auto)
            deltas.append(delta)
        preds = np.array(preds)
        deltas = np.array(deltas)
        rmses = np.sqrt(np.mean(deltas**2, axis=0))
        return rmses