# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from libs.dkf import BackwardLSTM, CombinerMLP, EncoderMLP, LatentSpaceTransitionMLP, DecoderMLP, Sampler

#--------------------------------------------------------------------
#
# Default parameters
#
#--------------------------------------------------------------------

X_DIM = 1 # Dimension of the observation space
Z_DIM = 16 # Dimension of the latent space
H_DIM = 16 # Dimension of the hidden state of the LSTM network(s)
G_DIM = 8 # Dimension of the output of the combiner
INTERMEDIATE_LAYER_DIM = 16 # Dimension of the intermediate layers of the MLPs

#----------------------------------------------------------------------
#
# Some Tests
#
#----------------------------------------------------------------------

def test_brick_1():
    """
    Test the backward LSTM brick.
    """
    input_size = 5
    hidden_size = 32
    num_layers = 3
        
    # Create the backward LSTM module
    lstm = BackwardLSTM(
        input_size=input_size, 
        hidden_size=hidden_size,
        num_layers=num_layers
    )
    # Create a random input tensor with shape (seq_len, batch_size, input_size)
    seq_len = 10
    batch_size = 4
    x = torch.randn(seq_len, batch_size, input_size)
    
    # Forward pass
    out = lstm(x)
    
    # report out
    
    print(f"Input shape: {x.shape}")  # should be (seq_len, batch_size, input_size)
    print(f"Backward LSTM output shape: {out.shape}")  # should be (seq_len, batch_size, hidden_size)

    assert out.shape == (seq_len, batch_size, hidden_size), f"Output shape {out.shape} is not equal to expected shape {(seq_len, batch_size, hidden_size)}"
    print("Backward LSTM test passed.")
    
def test_brick_2():
    """
    Test the combiner brick.
    """
    # First subtest -------------------------------------------------------------------------
    print(f"Test combiner 1")
    
    # Create the combiner module
    combiner = CombinerMLP(
        latent_dim=Z_DIM, 
        hidden_dim=H_DIM, 
        output_dim=G_DIM
    )
    print(f"Combiner summary :")
    print(combiner)
    
    # Create random input tensors
    batch_size = 16
    h = torch.randn(batch_size, H_DIM)  # hidden state of the backward LSTM at time t
    z = torch.randn(batch_size, Z_DIM)  # latent variable at time t-1
    
    # Forward pass
    g = combiner(h, z)
    
    # report out
    print(f"Input h shape: {h.shape}")  # should be (batch_size, H_DIM)
    print(f"Input z shape: {z.shape}")  # should be (batch_size, Z_DIM)
    print(f"Combiner output shape: {g.shape}")  # should be (batch_size, G_DIM)
    
    assert g.shape == (batch_size, G_DIM), f"Output shape {g.shape} is not equal to expected shape {(4, G_DIM)}"
    print("Combiner test 1 passed.")
    
    # Second subtest -------------------------------------------------------------------------
    print(f"Test combiner 2")
    hdim = 32
    zdim = 16
    gdim = 8
    
    # Create the combiner module
    combiner = CombinerMLP(
        latent_dim=zdim,
        hidden_dim=hdim,
        output_dim=gdim,
        layers_dim=[128, 128, 64, 32],
        activation='relu',
        inter_dim=64
    )
    print(f"Combiner summary :")
    print(combiner)
    
    # Create random input tensors
    batch_size = 16
    h = torch.randn(batch_size, hdim)  # hidden state of the backward LSTM at time t
    z = torch.randn(batch_size, zdim)  # latent variable at time t-1
    # Forward pass
    g = combiner(h, z)
    # report out
    print(f"Input h shape: {h.shape}")  # should be (batch_size, hdim)
    print(f"Input z shape: {z.shape}")  # should be (batch_size, zdim)
    print(f"Combiner output shape: {g.shape}")  # should be (batch_size, gdim)
    assert g.shape == (batch_size, gdim), f"Output shape {g.shape} is not equal to expected shape {(batch_size, gdim)}"
    print("Combiner test 2 passed.")
    
    
def test_brick_3():
    print(f"\nTesting EncoderMLP")
    
    # Create the encoder module
    zdim = 4
    gdim = 8
    layer_dim = [128, 128, 64, 32]
    
    encoder = EncoderMLP(
        latent_dim=zdim,
        combiner_dim=gdim,
        layers_dim=layer_dim,
        activation='tanh'
    )
    print(f"Encoder summary :")
    print(encoder)
    
    # Create random input tensors
    batch_size = 16
    g = torch.randn(batch_size, gdim)  # output of the combiner at time t
    # Forward pass
    mu, logvar = encoder(g)
    # report out
    print(f"Input g shape: {g.shape}")  # should be (batch_size, gdim)
    print(f"Encoder output mu shape: {mu.shape}")  # should be (batch_size, zdim)
    print(f"Encoder output logvar shape: {logvar.shape}")  # should be (batch_size, zdim)
    assert mu.shape == (batch_size, zdim), f"Output mu shape {mu.shape} is not equal to expected shape {(batch_size, zdim)}"
    assert logvar.shape == (batch_size, zdim), f"Output logvar shape {logvar.shape} is not equal to expected shape {(batch_size, zdim)}"
    print("EncoderMLP test passed.")