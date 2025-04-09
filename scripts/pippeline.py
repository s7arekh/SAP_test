import pandas as pd
import torch

from scripts.data_x18 import Datax18


device = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'mps' if torch.backends.mps.is_available()
    else 'cpu'
)
print(f"Using device: {device}")

df_theoretical = pd.read_csv(
    'data/theoretical_series.csv', 
    sep=';')
df_ssn = pd.read_csv(
    'data/SN_ms_tot_V2.0.csv',
    sep=';',
    header=None,
    names=['Year', 'Month', 'Decimal Date', 'SSN', 'Std Dev', 'Obs', 'Def'],
)
df_ssn = df_ssn[['Decimal Date', 'SSN']]
df_ssn = df_ssn[df_ssn['Decimal Date'] >= df_theoretical['decimal_date'].iloc[0]]

df_ssn = df_ssn[df_ssn['SSN'] > -1]

df_ssn = df_ssn.reset_index(drop=True)
theoretical_series = df_theoretical['T1'].values
observed_series = df_ssn['SSN'].values
prev_values = 4
horizon = 18
data = Datax18(theoretical_series, observed_series, prev_values, horizon)

train_data_X = data.X[:722 + 1 - (horizon - 1) - prev_values]
train_data_y = data.y[:722 + 1 - (horizon - 1) - prev_values]