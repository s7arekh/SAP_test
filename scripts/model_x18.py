import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NARXx18(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, M):
        self.M = M
        super(NARXx18, self).__init__()
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
        return self.net(x)
    
    def train_model(self, trainloader, test_data_X, test_data_y, epochs=1000, lr=0.001):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        train_losses, rmse_epochs = [], []

        # for epoch in tqdm(range(epochs), desc='Training', unit='epoch'):
        for epoch in range(epochs):
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
            
            # Считаем RMSE каждые 500 эпох
            if (epoch + 1) % 500 == 0:
                rmse = self.compute_rmse(test_data_X, test_data_y)
                rmse_epochs.append(rmse)
        return np.array(train_losses), np.array(rmse_epochs)
    
    def compute_rmse(self, test_data_X, test_data_y):
        self.eval()
        prediction = self(test_data_X.to(device)).squeeze()
        rmse = ((prediction.cpu().detach() - test_data_y).pow(2).mean(dim=0).sqrt() * self.M).numpy()
        return rmse