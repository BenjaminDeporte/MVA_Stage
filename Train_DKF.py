import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.dkf import DeepKalmanFilter, loss_function

def seed_everything(seed=42):
    """
    Set seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed for reproducibility
seed_everything(42)

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



n_steps = 50
n_series = 10000
s = generate_time_series(n_series, n_steps+1)



N = 3
fig, axs = plt.subplots(N, 1, figsize=(16, 3 * N))
for i in range(N):
    axs[i].plot(s[i], color='blue', marker="x", linewidth=1)
    axs[i].set_title(f"Time series {i+1}")
    axs[i].set_xlabel("Time")
    axs[i].set_ylabel("Value")
    axs[i].grid(True)
plt.tight_layout()
plt.show()


cutoff = int(0.8 * n_series)

X_train, y_train = s[:cutoff,:n_steps], s[:cutoff,-1]
X_valid, y_valid = s[cutoff:,:n_steps], s[cutoff:,-1]


# form datasets, dataloaders, etc

BATCH_SIZE = 8192

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

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)


xdim = 1
latent_dim = 16
h_dim = 16
combiner_dim = 4


dkf = DeepKalmanFilter(
    input_dim = xdim,
    latent_dim = latent_dim,
    hidden_dim = h_dim,
    combiner_dim = combiner_dim,
    num_layers = 1,
    device=device
).to(device)

# print(dkf)



optimizer = torch.optim.Adam(dkf.parameters(), lr=5e-4)
loss_fn = loss_function

# Training step : perform training for one epoch

def train_step(model, optimizer, criterion, train_loader=train_loader, device=device):
    ### training step
    model.train()
    optimizer.zero_grad()
    ### loop on training data
    rec_loss = 0
    kl_loss = 0
    epoch_loss = 0
    
    for input, _ in train_loader:
        input = input.to(device).unsqueeze(-1)  # add a feature dimension
        input = input.permute(1, 0, 2)  # permute to (seq_len, batch_size, input_dim)

        _, mu_x_s, logvar_x_s, mu_z_s, logvar_z_s, mu_z_transition_s, logvar_z_transition_s = model(input)
        
        rec_loss, kl_loss, total_loss = criterion(input, mu_x_s, logvar_x_s, mu_z_s, logvar_z_s, mu_z_transition_s, logvar_z_transition_s)
        
        total_loss.backward()
        optimizer.step()
              
        rec_loss += rec_loss.item()
        kl_loss += kl_loss.item()
        epoch_loss += total_loss.item()
        
    epoch_loss /= len(train_loader)
    rec_loss /= len(train_loader)
    kl_loss /= len(train_loader)
    
    return rec_loss, kl_loss, epoch_loss


num_epochs = 500

rec_losses = []
kl_losses = []
epoch_losses = []

for i in range(num_epochs):
    
    # run the training step
    rec_loss, kl_loss, epoch_loss = train_step(dkf, optimizer, loss_fn)
    # log results
    rec_losses.append(rec_loss)
    kl_losses.append(kl_loss)
    epoch_losses.append(epoch_loss)
    # Print the losses for this epoch
    if (i+1) % 10 == 0:
        print(f"Epoch {i+1:>3}/{num_epochs} - Rec Loss: {rec_loss:.4e}, KL Loss: {kl_loss:.4e}, Total Loss: {epoch_loss:.4e}")


# Plot the losses
plt.figure(figsize=(12, 6))
plt.plot(torch.tensor(rec_losses).cpu().detach(), label='Rec Loss', color='blue')
plt.plot(torch.tensor(kl_losses).cpu().detach(), label='KL Loss', color='orange')
plt.plot(torch.tensor(epoch_losses).cpu().detach(), label='Total Loss', color='green')
plt.title('Losses during training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()
