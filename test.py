import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from libs.vrnn_lib import ObservationLSTM, LatentLSTM, EncoderMLP, LatentStateTransitionMLP, DecoderMLP, VRNN

    
if __name__ == "__main__":

    print(f"Running Tests")
    
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
    
    x_dim = 2
    z_dim = 8
    rnn_z_hidden_dim = 16
    rnn_x_hidden_dim = 32
    
    vrnn = VRNN(
        input_dim=x_dim,
        latent_dim=z_dim,
        rnn_z_hidden_dim=rnn_z_hidden_dim,
        rnn_x_hidden_dim=rnn_x_hidden_dim
    )
    
    print(vrnn)