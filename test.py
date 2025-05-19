import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from libs.vrnn_lib import seed_everything, VRNN
from libs.vrnn_lib import loss_function, train
from libs.vrnn_lib import BetaLinearScheduler, BetaThresholdScheduler
from libs.vrnn_lib import plot_losses, sample_predictions

from libs.vrnn_lib import ObservationLSTM, LatentLSTM, EncoderMLP, LatentStateTransitionMLP, DecoderMLP, VRNN

    
if __name__ == "__main__":

    print(f"Running Tests")
    
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
        
    # # Test 1
    # seq_len = 50
    # batch_size = 32
    # x_dim = 10
    # h_dim = 20
    
    # bd = BidLSTM(input_size=x_dim, hidden_size=h_dim, num_layers=2)
   
    # x = torch.randn(seq_len, batch_size, x_dim)
    
    # out_f, out_b = bd(x)
    
    # print(f"Output shape: {out_f.shape}, {out_b.shape}")
    # # Check if the output shapes are correct
    
    # assert out_f.shape == (seq_len, batch_size, h_dim)
    # assert out_b.shape == (seq_len, batch_size, h_dim)
    
    # # test 2
    # print(bd)
    
    # # test 3
    # z_dim = 8
    # fl = ForwardLSTM(input_size=z_dim, hidden_size=h_dim, num_layers=2)
    
    # z = torch.randn(seq_len, batch_size, z_dim)
    # out = fl(z)
    
    # print(f"output shape: {out.shape}")
    # # Check if the output shape is correct
    # assert out.shape == (seq_len, batch_size, h_dim)
    
    # print(fl)
    
    # Test Encoder
    
    # z_dim = 8
    # rnn_z_hidden_dim = 16
    # rnn_x_hidden_dim = 32
    # layers_dim = [128,128,128]
    # batch_size = 4
    # seq_len = 50
    
    # enc = EncoderMLP(
    #     z_dim=z_dim,
    #     rnn_z_hidden_dim=rnn_z_hidden_dim,
    #     rnn_x_hidden_dim=rnn_x_hidden_dim,
    #     layers_dim=layers_dim
    # )
    # print(enc)
    
    # h = torch.randn(batch_size, rnn_z_hidden_dim)
    # g_fwd = torch.randn(batch_size, rnn_x_hidden_dim)
    # g_bwd = torch.randn(batch_size, rnn_x_hidden_dim)
    
    # mu, logvar = enc(h, g_fwd, g_bwd)
    # print(f"mu shape: {mu.shape}, logvar shape: {logvar.shape}")
    
    # Test Latent State Transition
    
    # z_dim = 8
    # rnn_z_hidden_dim = 16
    # rnn_x_hidden_dim = 32
    # layers_dim = [128,128,128]
    # batch_size = 4
    # seq_len = 50
    
    # lst = LatentStateTransitionMLP(
    #     z_dim=z_dim,
    #     rnn_z_hidden_dim=rnn_z_hidden_dim,
    #     rnn_x_hidden_dim=rnn_x_hidden_dim,
    #     layers_dim=layers_dim
    # )
    # print(lst)
    
    # h = torch.randn(batch_size, rnn_z_hidden_dim)
    # g_fwd = torch.randn(batch_size, rnn_x_hidden_dim)
    
    # mu, logvar = lst(h, g_fwd)
    # print(f"mu shape: {mu.shape}, logvar shape: {logvar.shape}")
    
    
    # Decoder
    
    # x_dim = 2
    # rnn_z_hidden_dim = 16
    # rnn_x_hidden_dim = 32
    # layers_dim = [128,128,128]
    # batch_size = 4
    # seq_len = 50

    
    # dec = DecoderMLP(
    #     x_dim=x_dim,
    #     rnn_z_hidden_dim=rnn_z_hidden_dim,
    #     rnn_x_hidden_dim=rnn_x_hidden_dim,
    #     layers_dim=layers_dim
    # )
    # print(dec)
    
    # h = torch.randn(batch_size, rnn_z_hidden_dim)
    # g_fwd = torch.randn(batch_size, rnn_x_hidden_dim)
    
    # mu, logvar = dec(h, g_fwd)
    # print(f"mu shape: {mu.shape}, logvar shape: {logvar.shape}")
    
    
    # test VRNN
    
    # x_dim = 2
    # z_dim = 8
    # rnn_z_hidden_dim = 16
    # rnn_x_hidden_dim = 32
    
    # vrnn = VRNN(
    #     input_dim=x_dim,
    #     latent_dim=z_dim,
    #     rnn_z_hidden_dim=rnn_z_hidden_dim,
    #     rnn_x_hidden_dim=rnn_x_hidden_dim
    # )
    
    # print(vrnn)
    
    # # test forward pass
    
    # seq_len = 50
    # batch_size = 4
    # x = torch.randn(seq_len, batch_size, x_dim)
    
    # x_t, mu_x_t, logvar_x_t, mu_phi_z_t, logvar_phi_z_t, mu_theta_z_t, logvar_theta_z_t = vrnn(x)
    # print(f"x_t shape: {x_t.shape}")
    # print(f"mu_x_t shape: {mu_x_t.shape}")
    # print(f"logvar_x_t shape: {logvar_x_t.shape}")
    # print(f"mu_phi_z_t shape: {mu_phi_z_t.shape}")
    # print(f"logvar_phi_z_t shape: {logvar_phi_z_t.shape}")
    # print(f"mu_theta_z_t shape: {mu_theta_z_t.shape}")
    # print(f"logvar_theta_z_t shape: {logvar_theta_z_t.shape}")
    
    
    
    X_DIM = 1 # Dimension of the observation space
    Z_DIM = 2 # Dimension of the latent space
    RNN_X_H_DIM = 4 # Dimension of the hidden state of the bidirectional LSTM network for observations
    RNN_Z_H_DIM = 8 # Dimension of the hidden state of the bidirectional LSTM network for latent variables
    
    n_steps = 50
    n_ahead = 10
    n_series = 100

    def generate_time_series(batch_size, n_steps, noise=0.05):
        """Utility function to generate time series data.

        Args:
            batch_size (int): number of time series to generate (btach size)
            n_steps (_type_): length of each time series
        """
        
        f1,f2,o1,o2 = np.random.rand(4, batch_size, 1)  # return 4 values for each time series
        time = np.linspace(0, 1, n_steps)  # time vector
        
        series = 0.4 * np.sin((time - o1) * (f1 * 5 + 10)) # first sine wave
        series += 0.2 * np.sin((time - o2) * (f2 * 20 + 20)) # second sine wave
        series += noise * (np.random.randn(batch_size, n_steps) - 0.5)  # add noise
        
        return series
    
    s = generate_time_series(n_series, n_steps+n_ahead)
    
    cutoff = int(0.8 * n_series)

    X_train, y_train = s[:cutoff,:n_steps], s[:cutoff,n_steps:]
    X_valid, y_valid = s[cutoff:,:n_steps], s[cutoff:,n_steps:]

    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_valid shape: {X_valid.shape}")
    print(f"y_valid shape: {y_valid.shape}")
    
    # form datasets, dataloaders, etc

    BATCH_SIZE = 16   # 8192 ok sur RTX3080 et 150 time steps

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
    test_dataset = TimeSeriesDataset(X_valid, y_valid)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    vrnn = VRNN(
        input_dim = X_DIM,
        latent_dim = Z_DIM,
        rnn_x_hidden_dim = RNN_X_H_DIM,
        rnn_z_hidden_dim = RNN_Z_H_DIM,
        device=device
    ).to(device)

    print(vrnn)
    
    optimizer = torch.optim.Adam(vrnn.parameters(), lr=1e-3)
    loss_fn = loss_function

    K = 1
    
    num_epochs = 5
    n_displays = 5
    display_frequency = int(num_epochs / n_displays)
    
    beta = 1e-3
    
    rec_losses, kl_losses, epoch_losses, val_rec_losses, val_kl_losses, val_epoch_losses, betas = train(
        vrnn, 
        optimizer, 
        loss_fn, 
        num_epochs=num_epochs, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        batch_size=BATCH_SIZE, 
        device=device, 
        beta=beta, 
        beta_scheduler=None, 
        display_frequency=display_frequency, 
        K=K
    )
    
    n_ahead = 10
    
    sample_predictions(N_SAMPLES=3, model=vrnn, X_valid=X_valid, y_valid=y_valid, n_steps=n_steps, n_ahead=n_ahead, device=device)