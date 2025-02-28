import numpy as np
import torch

class Data18Forward:
    def __init__(self,
                 model_series,
                 observed_series,
                 previous_values,
                 horizon=18):
        self.model_series = model_series
        self.observed_series = observed_series
        self.horizon = horizon
        self.n_prev = previous_values
        self.X, self.y = self._prepare_data()
        
    def _prepare_data(self):
        l = len(self.model_series) - self.n_prev - self.horizon + 1
        X = np.zeros((l, 2 * self.n_prev + self.horizon))
        y = np.zeros((l, self.horizon))
        for i in range(l):
            X[i, -self.n_prev:] = self.observed_series[i:i + self.n_prev]
            X[i, :-self.n_prev] = self.model_series[i:i + self.n_prev + self.horizon]
            y[i] = self.observed_series[i + self.n_prev: i + self.n_prev + self.horizon]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]