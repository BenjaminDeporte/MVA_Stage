#--------------------------------------------------------------------
#
# Deep Kalman Filter
#
#--------------------------------------------------------------------
#
# Deep Kalman Filter (DKF) implementation in PyTorch.
# Benjamin Deporte - Mai 2025
#
#---------------------------------------------------------------------


# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# set up
#--------------------------------------------------------------------

# Set the random seeds for reproducibility --------------------------

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

seed_everything(42)

# set device --------------------------------------------------------

if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = torch.device('cuda')
else:
    torch.set_default_device('cpu')
    device = torch.device('cpu')

print(f"Using {device}")

if device.type == 'cuda':
    print('GPU Name:', torch.cuda.get_device_name(0))
    print('Total GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
    
# set default tensor type ------------------------------------------

torch.set_default_dtype(torch.float64)

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

#--------------------------------------------------------------------
#
# Classes
#
#--------------------------------------------------------------------

#--- brick 1 : backward LSTM -----------------------------

class BackwardLSTM(nn.Module):
    """
    Backward LSTM module.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BackwardLSTM, self).__init__()
        
        self.input_size = input_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size,   # dimension of the observation space
            hidden_size,  # dimension of the hidden state of the LSTM network
            num_layers=num_layers, # number of layers of the LSTM network
            batch_first=False, # using the default PyTorch LSTM implementation, expecting input shape (seq_len, batch, input_size)
            bidirectional=False # unidirectional LSTM to start with
            )

    def forward(self, x):
        # Reverse the input sequence - axis 0 is the time axis here
        x_reversed = torch.flip(x, [0])
        # Pass through LSTM
        # using initial hidden state and cell state as zeros
        out, _ = self.lstm(x_reversed)
        # Reverse the output sequence
        out_reversed = torch.flip(out, [0])
        # return output shape (seq_len, batch, hidden_size)
        
        return out_reversed
    
#--- brick 2 : combiner -----------------------------
#
# this combines the latent variable at time t-1
# with the hidden state from the backward LSTM at time t,
# to compute a tensor g at time t, that will be used
# to compute the parameters of the approximate posterior distribution
# of the latent variable
#

class CombinerMLP(nn.Module):
    """Combiner module. Takes the hidden state of the backward LSTM at time t
    and the latent variable at time t-1, to compute a tensor g at time t,
    that will be used to compute the parameters of the approximate posterior
    distribution of the latent variable.

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, 
                 hidden_dim=H_DIM, 
                 output_dim=G_DIM,
                 layers_dim = None,  # list of layers dimensions, without the input dimnesion, without the output dimension
                 activation = 'tanh',
                 inter_dim = INTERMEDIATE_LAYER_DIM,
                 ):
        super(CombinerMLP, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported. Use 'tanh' or 'relu'.")
        self.inter_dim = inter_dim
        self.layers_dim = layers_dim
        
        if self.layers_dim is None:
            self.layers_dim = [inter_dim]
        else:
            self.layers_dim = layers_dim
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(self.layers_dim):
            if i==0:  #first layer, latent_dim + hidden_dim => layers_dim[0]
                layers.append(nn.Linear(latent_dim + hidden_dim, dim))
            else:  # all other layers
                layers.append(nn.Linear(self.layers_dim[i-1], dim))
            layers.append(self.activation_fn)
        # last layer : layers_dim[-1] => output_dim
        layers.append(nn.Linear(self.layers_dim[-1], output_dim))
        layers.append(self.activation_fn)
            
        # build the MLP
        self.mlp = nn.Sequential(*layers)
            
        
    def forward(self, h, z):
        """
        Forward pass of the combiner module.
        Args:
            h: hidden state of the backward LSTM at time t
            shape (batch, hidden_dim)
            z: latent variable at time t-1
            shape (batch, latent_dim)
        Returns:
            g: tensor g at time t
            shape (batch, output_dim)
        """
        
        # Concatenate the hidden state and the latent variable on their dimension
        x = torch.cat((h, z), dim=-1)
        
        # Pass through MLP
        g = self.mlp(x)
        
        return g     
        
#--- brick 3 : Encoder -----------------------------
#
# This computes the parameters of the approximate posterior distribution
# of the latent vatiable at time t.
# The approximate posterior distribution is a Gaussian distribution,
# we use a MLP to compute the mean and the log of the variance.
#

class EncoderMLP(nn.Module):
    """Encoder module. Computes the parameters of the approximate posterior
    distribution of the latent variable at time t. The approximate posterior
    distribution is a Gaussian distribution, we use a MLP to compute the mean
    and the log of the variance.

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, # Dimension of the latent space
                 combiner_dim=G_DIM, # Dimension of the combiner output
                 inter_dim=INTERMEDIATE_LAYER_DIM, # Dimension of the intermediate layers
                 layers_dim = None, # Dimension of the MLP layers (without inout nor output)
                 activation = 'tanh', # Activation function
    ):
        super(EncoderMLP, self).__init__()
        
        self.latent_dim = latent_dim
        self.combiner_dim = combiner_dim
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported. Use 'tanh' or 'relu'.")
        self.inter_dim = inter_dim
        
        if layers_dim is None:
            self.layers_dim = [self.inter_dim]
        else:
            self.layers_dim = layers_dim
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(self.layers_dim):
            if i==0:
                layers.append(nn.Linear(combiner_dim, dim))
            else:
                layers.append(nn.Linear(layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(layers_dim[-1], 2 * latent_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, g):
        """
        Forward pass of the encoder module.
        
        Args:
            g: tensor g at time t
            shape (batch, combiner_dim)
            
        Returns:
            mu: mean of the approximate posterior distribution
            shape (batch, latent_dim)
            logvar: log of the variance of the approximate posterior distribution
            shape (batch, latent_dim)
        """
        
        # Pass through MLP
        out = self.mlp(g)
        
        # Split the output into mean and log variance
        # each with shape (batch, latent_dim)
        mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        
        return mu, logvar
        
        
#--- brick 4 : Latent Space Transition -----------------------------       
#
# This computes the parameters of the transition distribution
# of the latent variable at time t. Ie the prior distribution, 
# before inference.
# The transition distribution is a Gaussian distribution,
# we use a MLP to compute the mean and the log of the variance.
#

class LatentSpaceTransitionMLP(nn.Module):
    """Latent space transition module. Computes the parameters of the
    transition distribution of the latent variable at time t. The transition
    distribution is a Gaussian distribution, we use a MLP to compute the mean
    and the log of the variance.

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, # Dimension of the latent space
                 inter_dim=INTERMEDIATE_LAYER_DIM, # Dimension of the intermediate layers
                 layers_dim = None, # Dimension of the MLP layers
                 activation = 'tanh', # Activation function
    ):
        super(LatentSpaceTransitionMLP, self).__init__()
        
        self.latent_dim = latent_dim
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported. Use 'tanh' or 'relu'.")
        self.inter_dim = inter_dim
        
        if layers_dim is None:
            layers_dim = [self.inter_dim]
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(layers_dim):
            if i==0:
                layers.append(nn.Linear(latent_dim, dim))
            else:
                layers.append(nn.Linear(layers_dim[i-1], dim))
            layers.append(self.activation)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(layers_dim[-1], 2 * latent_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
               
    def forward(self, z):
        """
        Forward pass of the latent space transition module.
        
        Args:
            z: latent variable at time t-1
            shape (batch, latent_dim)
            
        Returns:
            mu: mean of the transition distribution
            shape (batch, latent_dim)
            logvar: log of the variance of the transition distribution
            shape (batch, latent_dim)
        """
        
        # Pass through MLP
        out = self.mlp(z)
        
        # Split the output into mean and log variance
        # each with shape (batch, latent_dim)
        mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
        
        return mu, logvar
    
#--- brick 5 : Decoder (ie Observation Model) -----------------------------
#
# This computes the parameters of the distribution of 
# the observed variable 'x', given the latent variable 'z'.
# The distribution is a Gaussian distribution,
# we use a MLP to compute the mean and the log of the variance.
#

class DecoderMLP(nn.Module):
    """Decoder module. Computes the parameters of the distribution of the
    observed variable 'x', given the latent variable 'z'. The distribution is
    a Gaussian distribution, we use a MLP to compute the mean and the log of
    the variance.

    Args:
        nn (_type_): _description_
    
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, # Dimension of the latent space
                 observation_dim=X_DIM, # Dimension of the observation space
                 inter_dim=INTERMEDIATE_LAYER_DIM, # Dimension of the intermediate layers
                 layers_dim = None, # Dimension of the MLP layers
                 activation = 'tanh', # Activation function
    ):
        super(DecoderMLP, self).__init__()
        
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim
        self.inter_dim = inter_dim
        
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported. Use 'tanh' or 'relu'.")
        
        if layers_dim is None:
            layers_dim = [self.inter_dim] # one layer per default
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(layers_dim):
            if i==0:
                layers.append(nn.Linear(latent_dim, dim))
            else:
                layers.append(nn.Linear(layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(layers_dim[-1], 2 * self.observation_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Forward pass of the decoder module.
        Args:
            z: latent variable at time t
            shape (batch, latent_dim)
        Returns:
            mu: mean of the distribution of the observed variable
            shape (batch, observation_dim)
            logvar: log of the variance of the distribution of the observed variable
            shape (batch, observation_dim)
        """
        # Pass through MLP
        out = self.mlp(z)
        
        # Split the output into mean and log variance
        # each with shape (batch, observation_dim)
        mu, logvar = out[:, :self.observation_dim], out[:, self.observation_dim:]
        
        return mu, logvar
    
    
# --- brick 6 : Sampler with reparameterization trick -----------------------------
#
# This samples from a normal distribution of given mean and log variance
# using the reparameterization trick.

class Sampler(nn.Module):
    """Sampler module. Samples from a normal distribution of given mean and
    log variance using the reparameterization trick.

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self):
        super(Sampler, self).__init__()
        
    def forward(self, mu, logvar):
        """
        Forward pass of the sampler module.
        
        Args:
            mu: mean of the distribution
            shape (batch, dim)
            logvar: log of the variance of the distribution
            shape (batch, dim)
            
        Returns:
            v: sampled variables
            shape (batch, dim)
        """
        
        # Sample from a normal distribution using the reparameterization trick
        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)  # random noise
        v = mu + eps * std  # sampled variables
        
        return v
    


    