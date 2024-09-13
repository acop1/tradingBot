import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler

models = {}
device = "cpu"
batch_size = 16
lookback = 7
scaler = MinMaxScaler(feature_range=(-1,1))


def prepare_shifts(df, nsteps):
    df = dc(df)
    df.set_index('date', inplace=True)

    for i in range(nsteps, 0, -1):
        df[f'target(t-{i})'] = df['target'].shift(i)
    df.dropna(inplace=True)
    return df


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
def prep_lstm_data(symbol: str, today:str):
    path = '/Users/acop/Desktop/Projects/TradingBot/bot/StockCSV/FinHistData/Adjusted/'+ symbol +'adjFinHistData.csv'
    data = pd.read_csv(path)

    data = data[['date','close']]
    data = data.loc[(data['date'] < today)]
    test = data.copy().iloc[-1:]
    test.at[0, 'date'] = today
    test.at[0, 'close'] = 0
    test = test.drop(test[test['date'] != today].index)

    data['tomorrow'] = data['close'].shift(-1)
    test['tomorrow'] = test['close']
    data['target'] = (data['tomorrow'] > data['close']).astype(int)
    test['target'] = 0

    data = pd.concat([data, test], axis=0)
    data = data.reset_index()
    data = data.drop(columns={'index','close','tomorrow'})
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

    shiftedDf = prepare_shifts(data, lookback)

    shifted_df_as_np = shiftedDf.to_numpy()
    shifted_to_np = scaler.fit_transform(shifted_df_as_np)

    X = shifted_to_np[:, 1:]
    y = shifted_to_np[:, 0]

    split_index = int(len(X) -1)

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    X_train = X_train.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test, data
    

def predict_lstm(symbol: str):
    X_test = lstm_create(symbol)
    model = models[symbol+"_lstm"]
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback+1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies) 
    test_predictions = dc(dummies[:, 0])
    
    binary_pred = 10
    if test_predictions >= .6:
        binary_pred = 1
    else:
        binary_pred = 0

    return test_predictions, binary_pred

def lstm_create(symbol: str):
    train_loader, test_loader, X_test, data = prep_lstm_data(symbol, "2023-01-01")
    if (symbol+"_lstm") not in models.keys():

        model = LSTM(1, 4, 1)
        model.to(device)
        
        def train_one_epoch():
            model.train(True)
            running_loss = 0.0

            for batch_index, batch in enumerate(train_loader):
                x_batch, y_batch = batch[0].to(device), batch[1].to(device)

                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


        def validate_one_epoch():
                model.train(False)
                running_loss = 0.0

                for batch_index, batch in enumerate(test_loader):
                    x_batch, y_batch = batch[0].to(device), batch[1].to(device)

                    with torch.no_grad():
                        output = model(x_batch)
                        loss = loss_function(output, y_batch)
                        running_loss += loss.item()

        learning_rate = 0.01
        num_epochs = 10
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_one_epoch()
            validate_one_epoch()

        models[symbol+"_lstm"] = model
    return X_test
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#create a singular lstm model and use that model to predict. Do not train a seperate model for each day as that does not use machine learning. 
#completed this 