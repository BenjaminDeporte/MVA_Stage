import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.device('cuda')
    dtype = torch.cuda.FloatTensor
else:
    device = torch.device('cpu')
    dtype = torch.FloatTensor

print(f"Using {device}")

torch.set_default_dtype(torch.float64)

if device.type == 'cuda':
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Total GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
    


def generate_time_series(batch_size, n_steps, noise=0.05):
    """Utility function to generate time series data.

    Args:
        batch_size (int): number of time series to generate (btach size)
        n_steps (_type_): length of each time series
    """
    
    f1,f2,o1,o2 = np.random.rand(4, batch_size, 1)  # return 4 values for each time series
    time = np.linspace(0, 1, n_steps)  # time vector
    
    series = 0.8 * np.sin((time - o1) * (f1 * 10 + 10)) # first sine wave
    series += 0.2 * np.sin((time - o1) * (f1 * 20 + 20)) # second sine wave
    series += noise * (np.random.randn(batch_size, n_steps) - 0.5)  # add noise
    
    return series



n_steps = 150
n_series = 10000
s = generate_time_series(n_series, n_steps+1)



X_train, y_train = s[:7000,:n_steps], s[:7000,-1]
X_valid, y_valid = s[7000:9000,:n_steps], s[7000:9000,-1]
X_test, y_test = s[9000:,:n_steps], s[9000:,-1]


# form datasets, dataloaders, etc

from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X).to(device)
        self.y = torch.tensor(y).to(device)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = TimeSeriesDataset(X_train, y_train)
valid_dataset = TimeSeriesDataset(X_valid, y_valid)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)



# linear model

class LinearLayer(nn.Module):
    def __init__(self, n_steps, n_hidden=50):
        super(LinearLayer, self).__init__()
        self.fc1 = nn.Linear(n_steps, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
    
def train_linear_model(model,num_epochs=20,batch_size=32,lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    losses = []
    
    for i in range(num_epochs):
        epoch_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            # loss = F.mse_loss(outputs, targets.view(-1, 1))
            loss = torch.sqrt(torch.sum((outputs - targets.view(-1, 1)) ** 2))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_loader)
        losses.append(epoch_loss)
        if (i+1) % 10 == 0: 
            print(f"epoch {i+1}/{num_epochs}, training loss = {epoch_loss:4e}", end="\r")

    return losses



ll = LinearLayer(n_steps).to(device)

losses = train_linear_model(ll, num_epochs=5, batch_size=32, lr=1e-3)

y_pred = ll(torch.tensor(X_valid).to(device))
y_pred = y_pred.cpu().detach().numpy()
print(f"\n{np.mean(np.sqrt((y_valid - y_pred) ** 2)):.4f} RMSE")