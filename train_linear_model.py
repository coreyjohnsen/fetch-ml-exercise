import pandas as pd
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import os

data = pd.read_csv('data_daily.csv', sep=',', header=None)

new_data = []
for i in range(0,len(data)):
    date_obj = datetime.datetime.strptime(data.iloc[i,0], '%Y-%m-%d')
    entry = []
    entry.append(i)
    entry.append(data.iloc[i,1])
    new_data.append(entry)
init_data = data
data = pd.DataFrame(new_data, columns=['day', 'scans'])

shuffled_data = data.sample(frac=1)
train_cutoff = int(len(shuffled_data)*0.7)

X_train, X_test, y_train, y_test = shuffled_data.iloc[:train_cutoff,0], shuffled_data.iloc[train_cutoff:,0], shuffled_data.iloc[:train_cutoff,1], shuffled_data.iloc[train_cutoff:,1]

class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, x):
        out = self.fc(x)
        return out

model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.00002)

num_epochs = 400000

X_train, y_train = torch.Tensor(X_train.values).reshape((len(X_train), 1)), torch.Tensor(y_train.values).reshape((len(y_train), 1))

for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100000 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

X_test = torch.Tensor(X_test.values).reshape((len(X_test), 1))
with torch.no_grad():
    predictions = model(X_test)

print(f'Average error: {np.mean([abs(predictions[i].item() - y_test.iloc[i]) for i in range(len(y_test))])}')

if not os.path.exists('./models'):
    os.makedirs('./models')
torch.save(model.state_dict(), "models/fetch_2022_linear.pth")
