import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

class NARX18Forward(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, M):
        self.M = M
        super(NARX18Forward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Sigmoid(),
            *[
                layer
                for i in range(len(hidden_sizes) - 1)
                for layer in (nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]), nn.Sigmoid())
            ],
            nn.Linear(hidden_sizes[-1], output_size),
            nn.Softplus()
        )

    def forward(self, x):
        out = self.net(x)
        return out
    
    def train_model(self, trainloader, test_data_X, test_data_y, epochs=1000, lr=0.001):
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
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_losses.append(train_loss / len(trainloader))
            
            # Каждые 500 эпох считаем RMSE
            if (epoch + 1) % 500 == 0:
                rmse = self.compute_rmse(test_data_X, test_data_y)
                rmse_epochs.append(rmse)
        return np.array(train_losses), np.array(rmse_epochs)
    
    def compute_rmse(self, test_data_X, test_data_y):
        prediction = self(test_data_X.to(device)).squeeze()
        prediction = prediction.cpu().detach()
        rmse = ((prediction - test_data_y).pow(2).mean(dim=0).sqrt() * self.M).numpy()
        return rmse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')