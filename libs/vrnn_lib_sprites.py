#------------------------------------------------------------------
#
#
#------   MODELs, FUNCTIONS, CLASSES for Variationnal RNNs   ------
#
#------   APPLICATIONS TO SPRITES DATASET   -----------------------
#
#
#------------------------------------------------------------------

#--------------------------------------------------------------------
#
# Default parameters
#
#--------------------------------------------------------------------

# --- Image dimensions
H = 64
W = 64
C = 3

# --- Model parameters
X_DIM = 16 # Dimension of the observation space
Z_DIM = 16 # Dimension of the latent space
HIDDEN_Z_DIM = 16 # Dimension of the hidden state of the LSTM network(s) for Z
HIDDEN_X_DIM = 16 # Dimension of the hidden state of the LSTM network(s) for X
# G_DIM = 8 # Dimension of the output of the combiner - on va essayer de s'en passer
INTERMEDIATE_LAYER_DIM = 16 # Dimension of the intermediate layers of the MLPs

# --------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


#--------------------------------------------------------------------
#
#    Utilities
#
# --------------------------------------------------------------------

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
    
    
    
#--------------------------------------------------------------------
#
# CLASS VARIATIONNAL RNN
#
#--------------------------------------------------------------------

#--- brick 1 : Bidirectionnal LSTM for observations x_t -------------

class ObservationLSTM(nn.Module):
    """
    Bidirectionnal LSTM module.
    
    Creates a bidirectional LSTM, with num_layers layers, hidden state
    of dimension rnn_x_hidden_dim, and input of dimension input_size.
    The LSTM is used to process the input sequence in forward and backward order.
    """
    def __init__(self, input_size, rnn_x_hidden_size, num_layers=None):
        super(ObservationLSTM, self).__init__()
        
        self.input_size = int(input_size)
        if num_layers is None:
            self.num_layers = 1
        else:
            self.num_layers = int(num_layers)
        self.rnn_x_hidden_size = int(rnn_x_hidden_size)
        
        self.lstm = nn.LSTM(
            input_size = self.input_size,   # dimension of the observation space
            hidden_size = self.rnn_x_hidden_size,  # dimension of the hidden state of the LSTM network
            num_layers = self.num_layers, # number of layers of the LSTM network
            batch_first=False, # using the default PyTorch LSTM implementation, expecting input shape (seq_len, batch, input_size)
            bidirectional=True # bidirectional LSTM here
            )

    def forward(self, x):
        """
        "Forward" pass of the backward LSTM module.
        
        Args:
            x: input sequence
            shape (seq_len, batch, input_size)  - ASSUMING BATCH SECOND !
        Returns:
            g_fwd: output forward sequence - shape (seq_len, batch, hidden_size) - g_fwd[t] encodes x_{1:t}
            g_bwd: output backward sequence - shape (seq_len, batch, hidden_size) - g_bwd[t] encodes x_{t:T}
        """
        # sanity check
        assert self.input_size == x.shape[2], f"Input size {x.shape[2]} does not match the expected input size {self.input_size}"
        
        # Pass through LSTM using initial hidden state and cell state as zeros
        # x has shape (seq_len x batch x input_dim)
        out, _ = self.lstm(x) # shape (seq_len, batch, hidden_size * 2)
        
        # Split the output sequence in fwd and bwd halves
        g_fwd, g_bwd = out[:, :, :self.rnn_x_hidden_size], out[:, :, self.rnn_x_hidden_size:]
        
        # return output shape (seq_len, batch, hidden_size) x 2
        # g_fwd : forward LSTM output
        # g_bwd : backward LSTM output
        
        return g_fwd, g_bwd
    
    def __repr__(self):
        msg = f"Bidirectional LSTM (input_size={self.input_size}, rnn_x_hidden_size={self.rnn_x_hidden_size}, num_layers={self.num_layers})"
        return msg
    
#--- brick 2 : Forward LSTM for latent variable z_t -------------

class LatentLSTM(nn.Module):
    """
    Forward LSTM module for latent variables.
    
    Creates a forward LSTM, with num_layers layers, hidden state
    of dimension rnn_z_hidden_dim, and input of dimension input_size.
    The LSTM is used to process the input sequence of (sampled)
    latent variables in forward order.
    """
    def __init__(self, input_size, rnn_z_hidden_size, num_layers=None):
        super(LatentLSTM, self).__init__()
        
        self.input_size = int(input_size)
        if num_layers is None:
            self.num_layers = 1
        else:
            self.num_layers = int(num_layers)
        self.rnn_z_hidden_size = int(rnn_z_hidden_size)
        
        self.lstm = nn.LSTM(
            input_size = self.input_size,   # dimension of the observation space
            hidden_size = self.rnn_z_hidden_size,  # dimension of the hidden state of the LSTM network
            num_layers = self.num_layers, # number of layers of the LSTM network
            batch_first=False, # using the default PyTorch LSTM implementation, expecting input shape (seq_len, batch, input_size)
            bidirectional=False # unidirectional LSTM here
            )

    def forward(self, z):
        """
        "Forward" pass of the forward LSTM module.
        
        Args:
            z: input sequence of latent variables
            shape (seq_len, batch, z_dim)
            
        Returns:
            h: output sequence of hidden states. Hidden state out[t] encodes z_{1:t}
            shape (seq_len, batch, hidden_size)
        """
        
        # sanity check
        assert self.input_size == z.shape[2], f"Input size {z.shape[2]} does not match the expected input size {self.input_size}"
        
        # Pass through LSTM using initial hidden state and cell state as zeros
        h, _ = self.lstm(z) # shape (seq_len, batch, hidden_size)
        
        return h
    
    def __repr__(self):
        msg = f"Forward LSTM (input_size={self.input_size}, rnn_z_hidden_size={self.rnn_z_hidden_size}, num_layers={self.num_layers})"
        return msg


#--- brick 3 : Latent State Transition MLP, p_theta_z -----------------------------

class LatentStateTransitionMLP(nn.Module):
    """Latent State Transition module. 
    
    Computes the parameters of probability transition of latent variables.
     
    This transition distribution is a Gaussian distribution, we use a MLP to compute the mean
    and the log of the variance.
    
    The input variables of the latent state transition MLP are:
    - the tensor h of the Forward RNN for sampled latent variables z_t at time t-1. h[t-1] encodes z[1:t-1] - shape (batch_size x hidden_RNN_z)
    - the tensor g_fwd of the Bidirectional LSTM for observations x_t at time t-1. g_fwd[t-1] encodes x[1:t-1] - shape (batch_size x hidden_RNN_x)
    """
    
    default_num_units = 16 # Default dimension of the intermediate layers if no architecture provided
    
    def __init__(self, 
                 z_dim, # dimension of the latent space
                #  x_dim, # dimension of the observation space, unused
                 rnn_z_hidden_dim, # dimension of the hidden state of the Forward LSTM for latent variables
                 rnn_x_hidden_dim, # dimension of the hidden state of the Bidirectional LSTM for observations
                 default_num_units = None, # Dimension of the intermediate layers.
                 layers_dim = None, # list of number of neurons of the MLP layers.
                 activation = 'tanh', # Activation function
    ):
        """Creates the latent state transition module.
        This is a MLP, that takes as input:
            - the tensor h of the Forward RNN for sampled latent variables z_t at time t-1. h[t-1] encodes z[1:t-1] - shape (batch_size x hidden_RNN_z)
            - the tensor g_fwd of the Bidirectional LSTM for observations x_t at time t-1. g_fwd[t-1] encodes x[1:t-1] - shape (batch_size x hidden_RNN_x)

        The input dimension is hidden_rnn_z_dim + hidden_rnn_x_dim.
        
        The output dimension is the 2 * dimension of the latent space z_dim (mean, and log variance).
        The list of intermediate layers is passed in the list layers_dim.
        The activation function is passed as a string, either 'tanh' or 'relu'. (default = 'tanh')
        
        Inputs:
            z_dim : dimension of the latent space
            rnn_x_hidden_dim : dimension of the hidden state of the Bidirectional LSTM for observations
            rnn_z_hidden_dim : dimension of the hidden state of the Forward LSTM for latent variables
            default_num_units : dimension of the intermediate layers (default = self.default_num_units)
            layers_dim : list of layers dimensions, without the input dimension, without the output dimension
            activation : activation function (default = 'tanh')
        """
        super(LatentStateTransitionMLP, self).__init__()
        
        self.z_dim = z_dim
        self.rnn_z_hidden_dim = rnn_z_hidden_dim
        self.rnn_x_hidden_dim = rnn_x_hidden_dim

        # define activation
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported. Use 'tanh' or 'relu'.")
        
        # get architecture list ready
        if layers_dim is None:
            # if no architecture provided, create a MLP with 2 layers of default_num_units
            self.layers_dim = [self.default_num_units] * 2
        else:
            self.layers_dim = layers_dim
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(self.layers_dim):
            if i==0:
                # first layer, input dimension is hidden_rnn_z_dim + hidden_rnn_x_dim
                layers.append(nn.Linear(self.rnn_z_hidden_dim + self.rnn_x_hidden_dim, dim))
            else:
                layers.append(nn.Linear(self.layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(self.layers_dim[-1], 2 * self.z_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, h, g_fwd):
        """
        Forward pass of the latent state transition module.
        
        Takes as inputs:
            - the tensor h of the Forward RNN for sampled latent variables z_t at time t-1. h[t-1] encodes z[1:t-1] - shape (batch_size x hidden_RNN_z)
            - the tensor g_fwd of the Bidirectional LSTM for observations x_t at time t-1. g_fwd[t-1] encodes x[1:t-1] - shape (batch_size x hidden_RNN_x)

        The input dimension is hidden_rnn_z_dim + hidden_rnn_x_dim.
        
        The output dimension is the 2 * dimension of the latent space (mean, and log variance).

        The latent state transition module is used sequentially, so there is no seq_len dimension.
        
        The output is a tuple (mu, logvar), where mu is the mean of the approximate posterior distribution
        and logvar is the log of the variance of the approximate posterior distribution.
        
        Args:
            h: tensor h at time t-1 - shape (batch, rnn_z_hidden_dim)
            g_fwd: tensor g_fwd at time t-1 - shape (batch, rnn_x_hidden_dim)
            
        Returns:
            mu: mean of the approximate posterior distribution
            shape (batch, z_dim)
            logvar: log of the variance of the approximate posterior distribution
            shape (batch, z_dim)
        """
                   
        input = torch.cat((h, g_fwd), dim=1) # shape (batch, rnn_z_hidden_dim + rnn_x_hidden_dim)
        out = self.mlp(input) # shape (batch, 2 * self.z_dim)
        
        # Split the output into mean and log variance
        # each with shape (batch, latent_dim)
        mu, logvar = out[:, :self.z_dim], out[:, self.z_dim:]
        
        return mu, logvar
    
    def __repr__(self):
        msg = f"Latent State Transition MLP (z_dim={self.z_dim}, rnn_z_hidden_dim={self.rnn_z_hidden_dim}, rnn_x_hidden_dim={self.rnn_x_hidden_dim})"
        msg = msg + "\n" + self.mlp.__repr__()
        return msg
    

#--- brick 4 : Encoder q_phi -----------------------------
#
# This computes the parameters of the approximate posterior distribution
# of the latent vatiable at time t.
# The approximate posterior distribution is a Gaussian distribution,
# we use a MLP to compute the mean and the log of the variance.
#

class EncoderMLP(nn.Module):
    """Encoder module. q_phi.
    
    Computes the parameters of the approximate posterior
    distribution of the latent variable at time t.
     
    The approximate posterior distribution is a Gaussian distribution, we use a MLP to compute the mean
    and the log of the variance.
    
    The input variables of the encoder are:
    - the tensor h of the Forward RNN for sampled latent variables z_t at time t-1. - shape (batch_size x hidden_RNN_z) - NB: h[t-1] encodes z[1:t-1] 
    - the tensor g_fwd of the Bidirectional LSTM for observations x_t, at time t-1. - shape (batch_size x hidden_RNN_x) - NB: g_fwd[t-1] encodes x[1:t-1] 
    - the tensor g_bwd of the Bidirectional LSTM for observations x_t, at time t. - shape (batch_size x hidden_RNN_x) - NB: g_bwd[t] encodes x[t:T]
    """
    
    default_num_units = 16 # Default dimension of the intermediate layers if no architecture provided
    default_num_layers = 2 # Default number of layers if no architecture provided
    
    def __init__(self, 
                 z_dim, # dimension of the latent space
                #  x_dim, # dimension of the observation space, unused
                 rnn_z_hidden_dim, # dimension of the hidden state of the Forward LSTM for latent variables
                 rnn_x_hidden_dim, # dimension of the hidden state of the Bidirectional LSTM for observations
                 default_num_units = None, # Dimension of the intermediate layers.
                 layers_dim = None, # list of number of neurons of the MLP layers.
                 activation = 'tanh', # Activation function
    ):
        """Creates the encoder module.
        This is a MLP, that takes as input:
            - the tensor h of the Forward RNN for sampled latent variables z_t at time t-1. - shape (batch_size x hidden_RNN_z) - NB: h[t-1] encodes z[1:t-1] 
            - the tensor g_fwd of the Bidirectional LSTM for observations x_t, at time t-1. - shape (batch_size x hidden_RNN_x) - NB: g_fwd[t-1] encodes x[1:t-1] 
            - the tensor g_bwd of the Bidirectional LSTM for observations x_t, at time t. - shape (batch_size x hidden_RNN_x) - NB: g_bwd[t] encodes x[t:T]

        The input dimension is hidden_rnn_z_dim + 2 * hidden_rnn_x_dim.
        
        The output dimension is the 2 * dimension of the latent space z_dim (mean, and log variance).
        The list of intermediate layers is passed in the list layers_dim.
        The activation function is passed as a string, either 'tanh' or 'relu'. (default = 'tanh')
        
        Inputs:
            z_dim : dimension of the latent space
            rnn_x_hidden_dim : dimension of the hidden state of the Bidirectional LSTM for observations
            rnn_z_hidden_dim : dimension of the hidden state of the Forward LSTM for latent variables
            default_num_units : dimension of the intermediate layers (default = self.default_num_units)
            layers_dim : list of layers dimensions, without the input dimension, without the output dimension
            activation : activation function (default = 'tanh')
        """
        super(EncoderMLP, self).__init__()
        
        self.z_dim = int(z_dim)
        self.rnn_z_hidden_dim = int(rnn_z_hidden_dim)
        self.rnn_x_hidden_dim = int(rnn_x_hidden_dim)

        # define activation
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported. Use 'tanh' or 'relu'.")
        
        # get architecture list ready
        if layers_dim is None:
            # if no architecture provided, create a MLP with 2 layers of default_num_units
            self.layers_dim = [self.default_num_units] * self.default_num_layers
        else:
            self.layers_dim = layers_dim
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(self.layers_dim):
            if i==0:
                # first layer, input dimension is hidden_rnn_z_dim + 2 * hidden_rnn_x_dim
                layers.append(nn.Linear(self.rnn_z_hidden_dim + 2*self.rnn_x_hidden_dim, dim))
            else:
                layers.append(nn.Linear(self.layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(self.layers_dim[-1], 2 * self.z_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, h, g_fwd, g_bwd):
        """
        Forward pass of the encoder module.
        
        Takes as inputs:
            - the tensor h of the Forward RNN for sampled latent variables z_t at time t-1. - shape (batch_size x hidden_RNN_z) - NB: h[t-1] encodes z[1:t-1] 
            - the tensor g_fwd of the Bidirectional LSTM for observations x_t, at time t-1. - shape (batch_size x hidden_RNN_x) - NB: g_fwd[t-1] encodes x[1:t-1] 
            - the tensor g_bwd of the Bidirectional LSTM for observations x_t, at time t. - shape (batch_size x hidden_RNN_x) - NB: g_bwd[t] encodes x[t:T]

        The input dimension is hidden_rnn_z_dim + 2 * hidden_rnn_x_dim.
        
        The output dimension is the 2 * dimension of the latent space (mean, and log variance).

        The encoder module is used sequentially, so there is no seq_len dimension.
        
        The output is a tuple (mu, logvar), where mu is the mean of the approximate posterior distribution
        and logvar is the log of the variance of the approximate posterior distribution.
        
        Args:
            h: tensor h at time t-1 - shape (batch, rnn_z_hidden_dim)
            g_fwd: tensor g_fwd at time t-1 - shape (batch, rnn_x_hidden_dim)
            g_bwd: tensor g_bwd at time t - shape (batch, rnn_x_hidden_dim)
            
        Returns:
            mu: mean of the approximate posterior distribution
            shape (batch, z_dim)
            logvar: log of the variance of the approximate posterior distribution
            shape (batch, z_dim)
        """
                   
        input = torch.cat((h, g_fwd, g_bwd), dim=1) # shape (batch, rnn_z_hidden_dim + 2 * rnn_x_hidden_dim)
        out = self.mlp(input) # shape (batch, 2 * self.z_dim)
        
        # Split the output into mean and log variance
        # each with shape (batch, latent_dim)
        mu, logvar = out[:, :self.z_dim], out[:, self.z_dim:]
        
        return mu, logvar
    
    def __repr__(self):
        msg = f"Encoder MLP (z_dim={self.z_dim}, rnn_z_hidden_dim={self.rnn_z_hidden_dim}, rnn_x_hidden_dim={self.rnn_x_hidden_dim})"
        msg = msg + "\n" + self.mlp.__repr__()
        return msg
    
    
#--- brick 5 : Decoder aka Observation Model, p_theta_x -----------------------------

class DecoderMLP(nn.Module):
    """Decoder module. 
    
    Computes the parameters of the decoder, ie observation model.
     
    The observation model is a Gaussian distribution, we use a MLP to compute the mean
    and the log of the variance.
    
    The input variables of the latent state transition MLP are:
        - the tensor h of the Forward RNN for sampled latent variables z_t at time t. h[t] encodes z[1:t] - shape (batch_size x hidden_RNN_z)
        - the tensor g_fwd of the Bidirectional LSTM for observations x_t at time t-1. g_fwd[t-1] encodes x[1:t-1] - shape (batch_size x hidden_RNN_x)
    """
    
    default_num_units = 16 # Default dimension of the intermediate layers if no architecture provided
    default_num_layers = 2 # Default number of layers if no architecture provided
    
    def __init__(self, 
                #  z_dim, # dimension of the latent space, unused.
                 x_dim, # dimension of the observation space, unused
                 rnn_z_hidden_dim, # dimension of the hidden state of the Forward LSTM for latent variables
                 rnn_x_hidden_dim, # dimension of the hidden state of the Bidirectional LSTM for observations
                 default_num_units = None, # Dimension of the intermediate layers.
                 layers_dim = None, # list of number of neurons of the MLP layers.
                 activation = 'tanh', # Activation function
    ):
        """Creates the decoder module.
        This is a MLP, that takes as input:
            - the tensor h of the Forward RNN for sampled latent variables z_t at time t. h[t] encodes z[1:t] - shape (batch_size x hidden_RNN_z)
            - the tensor g_fwd of the Bidirectional LSTM for observations x_t at time t-1. g_fwd[t-1] encodes x[1:t-1] - shape (batch_size x hidden_RNN_x)

        The input dimension is hidden_rnn_z_dim + hidden_rnn_x_dim.
        
        The output dimension is the 2 * dimension of the observation space x_dim (mean, and log variance).
        The list of intermediate layers is passed in the list layers_dim.
        The activation function is passed as a string, either 'tanh' or 'relu'. (default = 'tanh')
        
        Inputs:
            x_dim : dimension of the latent space
            rnn_x_hidden_dim : dimension of the hidden state of the Bidirectional LSTM for observations
            rnn_z_hidden_dim : dimension of the hidden state of the Forward LSTM for latent variables
            default_num_units : dimension of the intermediate layers (default = self.default_num_units)
            layers_dim : list of layers dimensions, without the input dimension, without the output dimension
            activation : activation function (default = 'tanh')
        """
        super(DecoderMLP, self).__init__()
        
        self.x_dim = int(x_dim)
        self.rnn_z_hidden_dim = int(rnn_z_hidden_dim)
        self.rnn_x_hidden_dim = int(rnn_x_hidden_dim)

        # define activation
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'relu':
            self.activation_fn = nn.ReLU()
        else:
            raise ValueError(f"Activation function {activation} not supported. Use 'tanh' or 'relu'.")
        
        # get architecture list ready
        if layers_dim is None:
            # if no architecture provided, create a MLP with 2 layers of default_num_units
            self.layers_dim = [self.default_num_units] * self.default_num_layers
        else:
            self.layers_dim = layers_dim
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(self.layers_dim):
            if i==0:
                # first layer, input dimension is hidden_rnn_z_dim + hidden_rnn_x_dim
                layers.append(nn.Linear(self.rnn_z_hidden_dim + self.rnn_x_hidden_dim, dim))
            else:
                layers.append(nn.Linear(self.layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(self.layers_dim[-1], 2 * self.x_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, h, g_fwd):
        """
        Forward pass of the decoder module.
        
        Takes as inputs:
            - the tensor h of the Forward RNN for sampled latent variables z_t at time t. h[t] encodes z[1:t] - shape (batch_size x hidden_RNN_z)
            - the tensor g_fwd of the Bidirectional LSTM for observations x_t at time t-1. g_fwd[t-1] encodes x[1:t-1] - shape (batch_size x hidden_RNN_x)

        The input dimension is hidden_rnn_z_dim + hidden_rnn_x_dim.
        
        The output dimension is the 2 * dimension of the observation space (mean, and log variance).

        The latent state transition module is used sequentially, so there is no seq_len dimension.
        
        The output is a tuple (mu, logvar), where mu is the mean of the approximate posterior distribution
        and logvar is the log of the variance of the approximate posterior distribution.
        
        Args:
            h: tensor h at time t - shape (batch, rnn_z_hidden_dim)
            g_fwd: tensor g_fwd at time t-1 - shape (batch, rnn_x_hidden_dim)
            
        Returns:
            mu: mean of the approximate posterior distribution
            shape (batch, x_dim)
            logvar: log of the variance of the approximate posterior distribution
            shape (batch, x_dim)
        """
        
        input = torch.cat((h, g_fwd), dim=1) # shape (batch, rnn_z_hidden_dim + rnn_x_hidden_dim)
        out = self.mlp(input) # shape (batch, 2 * self.x_dim)
        
        # Split the output into mean and log variance
        # each with shape (batch, latent_dim)
        mu, logvar = out[:, :self.x_dim], out[:, self.x_dim:]
        
        return mu, logvar
    
    def __repr__(self):
        msg = f"Decoder MLP (x_dim={self.x_dim}, rnn_z_hidden_dim={self.rnn_z_hidden_dim}, rnn_x_hidden_dim={self.rnn_x_hidden_dim})"
        msg = msg + "\n" + self.mlp.__repr__()
        return msg

    
# --- brick 6 : Sampler with reparameterization trick -----------------------------
#
# This samples from a normal distribution of given mean and log variance
# using the reparameterization trick.

class Sampler(nn.Module):
    """Sampler module. Samples from a normal distribution of given mean and
    log variance using the reparameterization trick.
    
    NB : to be replaced by batch of tf.distributions.Normal(mu, logvar) ?
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
    

# --- brick 7 : CNN 3D Pre-Encoder ----------------------------------------------
#
# --- Takes (T, B) images of shape (W=64, H=64, C=3)
# --- pre-encodes them into (T, B, Dx)
# --- so they can be plugged as is in the VRNN model
#

class PreEncoderCNN(nn.Module):
    """
    Pre-Encoder CNN module.
    Takes images of shape (B, T, W=64, H=64, C=3)
    and encodes them into a latent space of dimension Dx.
    """
    def __init__(self, Dx=16):
        super(PreEncoderCNN, self).__init__()
        self.Dx = Dx  # Dimension of the output
        self.conv1 = nn.Conv2d(
            in_channels=3,  # input channels (C=3 for RGB images)
            out_channels=32,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,  # input channels from the previous layer
            out_channels=64,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,  # input channels from the previous layer
            out_channels=128,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, self.Dx)
    
    def forward(self, x):
        """
        Input: (T, B, W=64, H=64, C=3)
        Output: (T, B, Dx)
        """
        # check input shape
        assert x.dim() == 5, f"Input shape must be (T, B, W, H, C), got {x.shape}"
        assert x.shape[2] == 64 and x.shape[3] == 64 and x.shape[4] == 3, \
            f"Input shape must be (T, B, 64, 64, 3), got {x.shape}"
        # manage shape
        T = x.shape[0]  # Time dimension
        B = x.shape[1]  # Batch dimension
        W = x.shape[2]  # Width
        H = x.shape[3]  # Height
        C = x.shape[4]  # Channels
        # Reshape to (T*B, C, H, W) for CNN input
        x = x.reshape(T * B, W, H, C)  
        x = x.permute(0,3,2,1)  # (T*B,W,H,C) -> (T*B,C,H,W)
        # apply convolutional layers, pooling, etc
        x = self.conv1(x)  # (T*B, 32, 32, 32)
        x = F.relu(x)
        x = self.conv2(x)  # (T*B, 64, 16, 16)
        x = F.relu(x)
        x = self.conv3(x)  # (T*B, 128, 8, 8)
        x = F.relu(x)
        # # Flatten the output
        x = x.view(T * B, -1)  # (T*B, 128 * 8 * 8)
        # # Fully connected layers
        x = self.fc1(x)  # (T*B, 128)
        x = F.relu(x)
        x = self.fc2(x)  # (T*B, Dx)
        # Reshape back to (T, B, Dx)
        x = x.view(T, B, self.Dx)  # (T, B, Dx)

        return x
    
        
# --- brick 8 : CNN 3D Post-Decoder ----------------------------------------------
#
# --- Takes (B, T) outputs of shape (Dx)
# --- post-decodes them into (B, T, W=64, H=64, C=3)
# --- so they can be plugged as is in the VRNN model
#

class PostDecoderCNN(nn.Module):
    """
    post-Decoder CNN module.
    Takes outputs of shape (T, B, Dx)
    and decodes them into images of shape (T, B, W=64, H=64, C=3).
    NB: this is the reverse of the PreEncoderCNN.
    """
    def __init__(self, Dx=16):
        super(PostDecoderCNN, self).__init__()
        self.Dx = Dx  # Dimension of the input
        self.fc1 = nn.Linear(self.Dx, 128)
        self.fc2 = nn.Linear(128, 64 * 14 * 14)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,  # input channels (C=64)
            out_channels=64,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
            output_padding=1,  # to ensure output size is correct
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,  # input channels from the previous layer
            out_channels=32,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
            output_padding=1,  # to ensure output size is correct
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=32,  # input channels from the previous layer
            out_channels=3,  # output channels (C=3 for RGB images)
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
            output_padding=1,  # to ensure output size is correct
        )
        self.fc1 = nn.Linear(self.Dx, 128)
        self.fc2 = nn.Linear(128, 128 * 8 * 8)
    
    def forward(self, x):
        """
        Input: (T, B, Dx)
        Output: (T, B, W=64, H=64, C=3)
        """
        # check input shape
        assert x.dim() == 3, f"Input shape must be (T, B, Dx), got {x.shape}"
        assert x.shape[2] == self.Dx, f"Input shape must be (T, B, {self.Dx}), got {x.shape}"
        # fwd pass
        T = x.shape[0]  # Time dimension
        B = x.shape[1]  # Batch dimension
        # Reshape to (T*B, Dx) for fully connected layers
        x = x.reshape(T * B, self.Dx)  # (T*B, Dx)
        # Fully connected layers
        x = self.fc1(x)  # (T*B, 128)
        x = F.relu(x)
        x = self.fc2(x)  # (T*B, 128 * 8 * 8)
        x = F.relu(x)
        # Reshape to (T*B, 128, 8, 8)
        x = x.view(T * B, 128, 8 , 8)
        # Apply transposed convolutional layers, upsampling, etc
        x = self.deconv1(x)  # (T*B, 64, 16, 16)
        x = F.relu(x)
        x = self.deconv2(x)  # (T*B, 32, 32, 32)
        x = F.relu(x)
        x = self.deconv3(x)  # (T*B, 3, 64, 64)
        # # Reshape back to (T, B, W=64, H=64, C=3)
        x = x.view(T, B, 3, 64, 64)
        x = x.permute(0, 1, 3, 4, 2)  # (T, B, W=64, H=64, C=3)
        # output in [0,1] range
        x = torch.sigmoid(x)  # Ensure output is in [0, 1] range
        
        return x
        



#----------------------------------------------------------------
# NORMALIZED CNN PRE ENCODER AND POST DECODER FOR GPVAE
#----------------------------------------------------------------

# --- brick 7bis : NORMALIZED CNN 3D Pre-Encoder ----------------------------------------------
#
# --- Takes (T, B) images of shape (W=64, H=64, C=3)
# --- pre-encodes them into (T, B, Dx) with a Tanh activation at the end
# --- so they can be plugged as is in the VRNN model
#

class NormalizedPreEncoderCNN(nn.Module):
    """
    Pre-Encoder CNN module.
    Takes images of shape (B, T, W=64, H=64, C=3)
    and encodes them into a latent space of dimension Dx.
    """
    def __init__(self, Dx=16):
        super(NormalizedPreEncoderCNN, self).__init__()
        self.Dx = Dx  # Dimension of the output
        self.conv1 = nn.Conv2d(
            in_channels=3,  # input channels (C=3 for RGB images)
            out_channels=32,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=32,  # input channels from the previous layer
            out_channels=64,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=64,  # input channels from the previous layer
            out_channels=128,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
        )
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, self.Dx)
    
    def forward(self, x):
        """
        Input: (T, B, W=64, H=64, C=3)
        Output: (T, B, Dx)
        """
        # check input shape
        assert x.dim() == 5, f"Input shape must be (T, B, W, H, C), got {x.shape}"
        assert x.shape[2] == 64 and x.shape[3] == 64 and x.shape[4] == 3, \
            f"Input shape must be (T, B, 64, 64, 3), got {x.shape}"
        # manage shape
        T = x.shape[0]  # Time dimension
        B = x.shape[1]  # Batch dimension
        W = x.shape[2]  # Width
        H = x.shape[3]  # Height
        C = x.shape[4]  # Channels
        # Reshape to (T*B, C, H, W) for CNN input
        x = x.reshape(T * B, W, H, C)  
        x = x.permute(0,3,2,1)  # (T*B,W,H,C) -> (T*B,C,H,W)
        # apply convolutional layers, pooling, etc
        x = self.conv1(x)  # (T*B, 32, 32, 32)
        x = F.relu(x)
        x = self.conv2(x)  # (T*B, 64, 16, 16)
        x = F.relu(x)
        x = self.conv3(x)  # (T*B, 128, 8, 8)
        x = F.tanh(x)
        # # Flatten the output
        x = x.view(T * B, -1)  # (T*B, 128 * 8 * 8)
        # # Fully connected layers
        x = self.fc1(x)  # (T*B, 128)
        x = F.relu(x)
        x = self.fc2(x)  # (T*B, Dx)
        # Reshape back to (T, B, Dx)
        x = x.view(T, B, self.Dx)  # (T, B, Dx)

        return x
    
        
# --- brick 8 : CNN 3D Normalized Post-Decoder ----------------------------------------------
#
# --- Takes (B, T) outputs of shape (Dx)
# --- post-decodes them into (B, T, W=64, H=64, C=3)
# --- so they can be plugged as is in the VRNN model
#

class NormalizedPostDecoderCNN(nn.Module):
    """
    post-Decoder CNN module.
    Takes outputs of shape (T, B, Dx)
    and decodes them into images of shape (T, B, W=64, H=64, C=3).
    NB: this is the reverse of the PreEncoderCNN.
    """
    def __init__(self, Dx=16):
        super(NormalizedPostDecoderCNN, self).__init__()
        self.Dx = Dx  # Dimension of the input
        self.fc1 = nn.Linear(self.Dx, 128)
        self.fc2 = nn.Linear(128, 64 * 14 * 14)
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=128,  # input channels (C=64)
            out_channels=64,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
            output_padding=1,  # to ensure output size is correct
        )
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=64,  # input channels from the previous layer
            out_channels=32,
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
            output_padding=1,  # to ensure output size is correct
        )
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=32,  # input channels from the previous layer
            out_channels=3,  # output channels (C=3 for RGB images)
            kernel_size=3,  # size of the convolutional kernel
            stride=2,  # stride of the convolution
            padding=1,
            output_padding=1,  # to ensure output size is correct
        )
        self.fc1 = nn.Linear(self.Dx, 128)
        self.fc2 = nn.Linear(128, 128 * 8 * 8)
    
    def forward(self, x):
        """
        Input: (T, B, Dx)
        Output: (T, B, W=64, H=64, C=3)
        """
        # check input shape
        assert x.dim() == 3, f"Input shape must be (T, B, Dx), got {x.shape}"
        assert x.shape[2] == self.Dx, f"Input shape must be (T, B, {self.Dx}), got {x.shape}"
        # fwd pass
        T = x.shape[0]  # Time dimension
        B = x.shape[1]  # Batch dimension
        # Reshape to (T*B, Dx) for fully connected layers
        x = x.reshape(T * B, self.Dx)  # (T*B, Dx)
        # Fully connected layers
        x = self.fc1(x)  # (T*B, 128)
        x = F.relu(x)
        x = self.fc2(x)  # (T*B, 128 * 8 * 8)
        x = F.relu(x)
        # Reshape to (T*B, 128, 8, 8)
        x = x.view(T * B, 128, 8 , 8)
        # Apply transposed convolutional layers, upsampling, etc
        x = self.deconv1(x)  # (T*B, 64, 16, 16)
        x = F.relu(x)
        x = self.deconv2(x)  # (T*B, 32, 32, 32)
        x = F.relu(x)
        x = self.deconv3(x)  # (T*B, 3, 64, 64)
        # # Reshape back to (T, B, W=64, H=64, C=3)
        x = x.view(T, B, 3, 64, 64)
        x = x.permute(0, 1, 3, 4, 2)  # (T, B, W=64, H=64, C=3)
        # output in [0,1] range
        x = torch.sigmoid(x)  # Ensure output is in [0, 1] range
        
        return x


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#
# --- Class VRNN : Variational RNN --------------------------------
#
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------


class VRNN(nn.Module):
    """Variational RNN module. Implement the VRNN model.
    
    Args:
        nn (_type_): _description_
        
    Returns:
        _type_: _description_
    """
    
    def __init__(self,
                 input_dim, # Dimension of the observation space
                 latent_dim, # Dimension of the latent space
                 rnn_x_hidden_dim, # Dimension of the hidden state of the LSTM network for observations
                 rnn_z_hidden_dim, # Dimension of the hidden state of the LSTM network for latent variables
                 rnn_num_layers=1, # Number of layers of the LSTM networks
                 inter_dim=None, # Dimension of the intermediate layers
                 layers_dim_encoder = None, # list of layers dimensions for the encoder module
                 layers_dim_transition = None, # list of layers dimensions for the transition module
                 layers_dim_decoder = None, # list of layers dimensions for the decoder module
                 activation='tanh', # Activation function
                 device='cpu' # Device to use (cpu or cuda)
                 ):
        super(VRNN, self).__init__()
        
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.rnn_x_hidden_dim = int(rnn_x_hidden_dim)
        self.rnn_z_hidden_dim = int(rnn_z_hidden_dim)
        self.inter_dim = inter_dim
        self.activation = activation
        self.layers_dim_encoder = layers_dim_encoder
        self.layers_dim_transition = layers_dim_transition
        self.layers_dim_decoder = layers_dim_decoder
        self.device = device
        
        # define the modules
        
        self.observation_lstm = ObservationLSTM(
            input_size=self.input_dim,
            rnn_x_hidden_size=self.rnn_x_hidden_dim,
            num_layers=rnn_num_layers
        )
        
        self.latent_lstm = LatentLSTM(
            input_size=self.latent_dim,
            rnn_z_hidden_size=self.rnn_z_hidden_dim,
            num_layers=rnn_num_layers
        )
        
        self.encoder = EncoderMLP(
            z_dim=self.latent_dim,
            rnn_z_hidden_dim=self.rnn_z_hidden_dim,
            rnn_x_hidden_dim=self.rnn_x_hidden_dim,
            default_num_units=self.inter_dim,
            layers_dim=self.layers_dim_encoder, # list of layers dimensions, without the input dimension, without the output dimension
            activation=self.activation,
        )        
            
        self.latent_state_transition = LatentStateTransitionMLP(
            z_dim = self.latent_dim,
            rnn_z_hidden_dim = self.rnn_z_hidden_dim,
            rnn_x_hidden_dim = self.rnn_x_hidden_dim,
            default_num_units=self.inter_dim,
            layers_dim=self.layers_dim_transition, # list of layers dimensions, without the input dimension, without the output dimension
            activation=self.activation,
        )
        
        self.decoder = DecoderMLP(
            x_dim=self.input_dim,
            rnn_z_hidden_dim=self.rnn_z_hidden_dim,
            rnn_x_hidden_dim=self.rnn_x_hidden_dim,
            default_num_units=self.inter_dim,
            layers_dim=self.layers_dim_decoder, # list of layers dimensions, without the input dimension, without the output dimension
            activation=self.activation,
        )
                
        self.sampler = Sampler()
        
    def __repr__(self):
        
        msg = f"VRNN" +f"(observation_dim={self.input_dim}, latent_dim={self.latent_dim})"
        msg += f"\n{self.observation_lstm}"
        msg += f"\n{self.latent_lstm}"
        msg += f"\n{self.encoder}"
        msg += f"\n{self.latent_state_transition}"
        msg += f"\n{self.decoder}"
        msg += f"\n{self.sampler}"
        
        return msg

    def forward(self, x_t):
        """
        Forward pass of the Variational RNN.
        
        Runs one step inference :
        
        0- Initialization are run (sampled latent variables of sequence length)
        1- The input sequence x_t (seq_len, batch, input_dimension) is passed through the bidirectional observation LSTM 
           to get the hidden states g_fwd and g_bwd (seq_len, batch, rnn_x_hidden_dim).
        2- A sequential loop is run from time t=1 to t=T (seq_len):
            2.1- the sampled latent variables tensor, up to previous time step z_samples[1:t-1], is run throught the forward
                latent state LSTM, to get the hidden state h[t-1] (encoding z[1:t-1]).
            2.2- ENCODER : the tensor h[t-1] (encoding z[1:t-1]), the tensor g_fwd[t-1] (encoding x[1:t-1]), and the tensor g_bwd[t] 
                (encoding x[t:T]) are passed through the encoder module to get the parameters of the approximate posterior 
                distribution mu_phi and logvar_phi of the latent variable z_t.
            2.3- TRANSITION PRIOR : the tensor h[t-1] (encoding z[1:t-1]) and the tensor g_fwd[t-1] (encoding x[1:t-1]) are
                passed through the transition module to get the parameters of the transition distribution of the 
                latent variable z_t, mu_theta_z and logvar_theta_z.
            2.4- the latent variable z_t at time t is sampled from the approximate posterior distribution and stored
                into the sequence of sampled latent variables z_samples at index t.
            2.5- the sampled latent variables tensor, now updated with sample z[t], is passed through the forward
                latent state LSTM to get the hidden state h[t] (encoding z[1:t]).
            2.6- DECODER : the tensor h[t] (encoding z[1:t]), and the tensor g_fwd[t-1] (encoding x[1:t-1]) are passed 
                through the decoder module to get the parameters of the distribution of the observed variables x_t,
                mu_x_t and logvar_x_t. 

        Args:
            x_t: input sequence - shape (seq_len, batch, input_dim)
            
        Intermediate variables:
            sampled_z_t : sequence of sampled latent variables - shape (seq_len, batch, latent_dim).
            NB : z_0 is set to 0.
            g_fwd : hidden states of the forward pass of the bidirectional observation LSTM - shape (seq_len, batch, rnn_x_hidden_dim)
            g_bwd : hidden states of the backward pass of the bidirectional observation LSTM - shape (seq_len, batch, rnn_x_hidden_dim)
            h_t : hidden states of the forward latent state LSTM - shape (seq_len, batch, rnn_z_hidden_dim)
        
        Returns:
            mu_x_t: means of the distribution of the observed variables - shape (seq_len, batch, input_dim)
            logvar_x_t: log of the variances of the distribution of the observed variables - shape (seq_len, batch, input_dim)
            mu_phi_z_t: means of the approximate posterior distribution (q_\phi) of the latent variable - shape (seq_len, batch, latent_dim)
            logvar_phi_z_t: log of the variances of the approximate posterior distribution (q_\phi) of the latent variable - shape (seq_len, batch, latent_dim)
            mu_theta_z_t: means of the transition distribution (p_\theta_z) of the latent variable - shape (seq_len, batch, latent_dim)
            logvar_theta_z_t: log of the variances of the transition distribution (p_\theta_z) of the latent variable - shape (seq_len, batch, latent_dim)
        """
        
       
        # we assume that the input sequence is of shape (seq_len, batch, input_dim) and check some
        seq_len, batch_size, input_dim = x_t.shape
        assert input_dim == self.input_dim, f"Input dimension {input_dim} does not match the expected dimension {self.input_dim}"
        
        # initializations of the tensors for the sequential loop
                # NB : in INRIA code : self.register_buffer
                # "If you have parameters in your model, which should be saved and restored in the state_dict, 
                # but not trained by the optimizer, you should register them as buffers.
                # Buffers won’t be returned in model.parameters(), 
                # so that the optimizer won’t have a change to update them.#
                
        # sampled latent variables tensor, shape (seq_len, batch_size, latent_dim)
        sampled_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        z0 = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)  # initial latent variable z_0
        g0 = torch.zeros(1, batch_size, self.rnn_x_hidden_dim).to(self.device)  # initial hidden state of the observation LSTM
        
        # place holders for the parameters of the distributions
        mu_phi_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        logvar_phi_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        mu_theta_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        logvar_theta_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        mu_x_t = torch.zeros(seq_len, batch_size, self.input_dim).to(self.device)
        logvar_x_t = torch.zeros(seq_len, batch_size, self.input_dim).to(self.device)
        
        # step 1 : run the bidirectional observation LSTM on the input sequence
        g_fwd, g_bwd = self.observation_lstm(x_t) # shape (seq_len, batch_size, rnn_x_hidden_dim) each
        
        # step 2 : loop from t=0 to t=T-1 (T = seq_len) for the sequential logic
        for t in range(seq_len):
            
            # step 2.1 : at time t, we need the value h[t-1] of the fwd RNN of the latent variables (encoding z[1:t-1]))
            if t == 0:
                # at time t=0, we use the initial value z_0=0
                h_t = self.latent_lstm(z0) # shape (seq_len, batch_size, rnn_z_hidden_dim)
                h_t_minus_1 = h_t[0,:,:] # shape (batch_size, rnn_z_hidden_dim)
                g_fwd_t_minus_1 = g0.squeeze(0) # shape (batch_size, rnn_x_hidden_dim)
            else:
                # when t>=1, we run the whole sequence of sampled latent variables
                # into the latent state LSTM to get the hidden state h[t-1]
                # h_t = self.latent_lstm(sampled_z_t) has been done at the end of the previous loop
                h_t_minus_1 = h_t[t-1,:,:] # shape (batch_size, rnn_z_hidden_dim)
                g_fwd_t_minus_1 = g_fwd[t-1,:,:] # shape (batch_size, rnn_x_hidden_dim)
                
            # step 2.2 : ENCODER : h[t-1] and g_fwd[t-1], g_bwd[t] are passed through the encoder module
            mu_phi, logvar_phi = self.encoder(h_t_minus_1, g_fwd_t_minus_1, g_bwd[t,:,:])
            mu_phi_z_t[t], logvar_phi_z_t[t] = mu_phi, logvar_phi
            
            # step 2.3 : TRANSITION PRIOR : h[t-1] and g_fwd[t-1] are passed through the transition module
            mu_theta_z, logvar_theta_z = self.latent_state_transition(h_t_minus_1, g_fwd_t_minus_1)
            mu_theta_z_t[t], logvar_theta_z_t[t] = mu_theta_z, logvar_theta_z
            
            # step 2.4 : sample z_t from the approximate posterior distribution
            # NB : here, cloning is used to avoid in-place operation, that would break the computation graph
            # and raise a runtime error when computing the gradients !
            temp = sampled_z_t.clone()
            temp[t,:,:] = self.sampler(mu_phi, logvar_phi)
            sampled_z_t = temp
            
            # step 2.5 : update the output of the forward latent state LSTM with the new sampled latent variable
            h_t = self.latent_lstm(sampled_z_t) # ( seq_len, batch_size, rnn_z_hidden_dim)
            
            # step 2.6 : DECODER : h[t] and g_fwd[t-1] are passed through the decoder module
            mu_x, logvar_x = self.decoder(h_t[t,:,:], g_fwd_t_minus_1)
            mu_x_t[t], logvar_x_t[t] = mu_x, logvar_x
                                   
        # return the outputs
        return x_t, mu_x_t, logvar_x_t, mu_phi_z_t, logvar_phi_z_t, mu_theta_z_t, logvar_theta_z_t
    

    
    def predict(self, x, num_steps):
        """
        Predicts future steps based on the input sequence.

        Args:
            x_t (torch.Tensor): Input tensor of shape (seq_len, batch_size, x_dim).
            num_steps (int): Number of future steps to predict.

        Returns:
            mu_predictions (torch.Tensor): Tensor of shape (num_steps, batch_size, x_dim)
            containing the means of the predicted observations at future steps.
            logvar_predictions (torch.Tensor): Tensor of shape (num_steps, batch_size, x_dim)
            containing the log variances of the predicted observations at future steps.
            
            mu_full_x (torch.Tensor): Tensor of shape (seq_len + num_steps, batch_size, x_dim)
            containing the reconstructed input sequence and the means of the predicted future steps.
            logvar_full_x (torch.Tensor): Tensor of shape (seq_len + num_steps, batch_size, x_dim)
            containing the log variances of the reconstructed input sequence and the predicted future steps.
        """
        
        with torch.no_grad():
            
            # get hyperparameters
            seq_len, batch_size, input_dim = x.shape
            assert input_dim == self.input_dim, f"Input dimension {input_dim} does not match the expected dimension {self.input_dim}"
            
            # run an inference forward pass to get the parameters of the observation distribution,
            # and the transition distribution of the latent variable
            x_t, mu_x_t, logvar_x_t, mu_phi_z_t, logvar_phi_z_t, mu_theta_z_t, logvar_theta_z_t = self.forward(x)

            # Initialize predictions
            mu_x_predictions = torch.zeros(num_steps, batch_size, self.input_dim).to(self.device)
            logvar_x_predictions = torch.zeros(num_steps, batch_size, self.input_dim).to(self.device)
            mu_theta_z_predictions = torch.zeros(num_steps, batch_size, self.latent_dim).to(self.device)
            logvar_theta_z_predictions = torch.zeros(num_steps, batch_size, self.latent_dim).to(self.device)
            
            # Initialize the whole sequence (input + predictions)
            mu_full_x = torch.cat([mu_x_t, mu_x_predictions], dim=0)
            logvar_full_x = torch.cat([logvar_x_t, logvar_x_predictions], dim=0)
            mu_full_theta_z = torch.cat([mu_theta_z_t, mu_theta_z_predictions], dim=0)
            logvar_full_theta_z = torch.cat([logvar_theta_z_t, logvar_theta_z_predictions], dim=0)
            
            # compute predicted z_t and x_t with autoregression
            for s in range(num_steps):
                
                # first, compute h[t-1] with the last sampled latent variable
                h_t = self.latent_lstm(mu_full_theta_z) # shape (seq_len + num_steps, batch_size, rnn_z_hidden_dim)
                h_t_minus_1 = h_t[seq_len + s-1,:,:] # shape (batch_size, rnn_z_hidden_dim)
                # second, get the g_fwd[t-1] from the observation LSTM
                g_fwd, _ = self.observation_lstm(mu_full_x) # shape (seq_len + num_steps, batch_size, rnn_x_hidden_dim)
                g_fwd_t_minus_1 = g_fwd[seq_len + s-1,:,:] # shape (batch_size, rnn_x_hidden_dim)
                
                # predict parameters of z_t at the current time step
                mu_theta_z, logvar_theta_z = self.latent_state_transition(h_t_minus_1, g_fwd_t_minus_1)
                
                # sample z_t from the transition distribution
                sampled_z_t = self.sampler(mu_theta_z, logvar_theta_z)
                mu_full_theta_z[seq_len + s], logvar_full_theta_z[seq_len + s] = sampled_z_t, logvar_theta_z
                
                # update h_t with the predicted z_t
                h_t = self.latent_lstm(mu_full_theta_z) # shape (seq_len + num_steps, batch_size, rnn_z_hidden_dim)
                # get the parameters of the predicted observation distribution
                mu_x_pred, logvar_x_pred = self.decoder(h_t[seq_len + s,:,:], g_fwd_t_minus_1)
                
                # store those parameters
                mu_x_predictions[s,:,:] = mu_x_pred
                mu_full_x[seq_len + s,:,:] = mu_x_pred
                logvar_x_predictions[s,:,:] = logvar_x_pred
                logvar_full_x[seq_len + s,:,:] = logvar_x_pred
        
        return mu_x_predictions, logvar_x_predictions, mu_full_x, logvar_full_x


#----------------------------------------------------------------------
#
# Loss Function
#
#----------------------------------------------------------------------


def loss_function(x_t, mu_x_t, logvar_x_t, mu_phi_z_t, logvar_phi_z_t, mu_theta_z_t, logvar_theta_z_t, beta=None):
    # """
    # Version 2.0 of the loss function for the Deep Kalman Filter.
    # All distributions parameters are assumed to have been computed from K samples of each z_t.
    # The K samples are used to compute the expectations within the loss function.

    # Args:
    #     x_t (tensor): the original input sequence - shape (seq_len, batch_size, x_dim)

    #     mu_x_t (tensor): the mean of the distribution of the observed variable - shape (seq_len, batch_size, x_dim, K)
    #     logvar_x_t (tensor): the log of the variance of the distribution of the observed variable - shape (seq_len, batch_size, x_dim, K)
    #     mu_phi_z_t (tensor): the mean of the approximate posterior distribution (q_\phi) of the latent variable - shape (seq_len, batch_size, z_dim, K)
    #     logvar_phi_z_t (tensor): the log of the variance of the approximate posterior distribution (q_\phi) of the latent variable - shape (seq_len, batch_size, z_dim, K)
    #     mu_theta_z_t (tensor): the mean of the transition distribution (p_\theta_z) of the latent variable - shape (seq_len, batch_size, z_dim, K)
    #     logvar_theta_z_t (tensor): the log of the variance of the transition distribution (p_\theta_z) of the latent variable - shape (seq_len, batch_size, z_dim, K)

    #     beta (float, optional): the weight of the KL divergence term in the loss function. Defaults to None.
        
    # Returns:
    #     rec_loss (tensor): the reconstruction loss - shape (1,)
    #     kl_loss (tensor): the KL divergence loss - shape (1,)
    #     total_loss (tensor): the total loss - shape (1,)
    # """
    
    seq_len, batch_size, x_dim = x_t.shape
    
    # choose beta
    if beta is None:
        beta = 1.0
        
    # check whether there is a K dimension or not, add K=1 if none given
    if mu_x_t.dim() == 3:
        K = 1
        mu_x_t = mu_x_t.unsqueeze(-1)
        logvar_x_t = logvar_x_t.unsqueeze(-1)
        mu_phi_z_t = mu_phi_z_t.unsqueeze(-1)
        logvar_phi_z_t = logvar_phi_z_t.unsqueeze(-1)
        mu_theta_z_t = mu_theta_z_t.unsqueeze(-1)
        logvar_theta_z_t = logvar_theta_z_t.unsqueeze(-1)
    else:
        K = mu_x_t.shape[-1]
        
    z_dim = mu_phi_z_t.shape[-2]
        
    # compute the expectation of the reconstruction loss with K samples
    
    x_t_extended = x_t.unsqueeze(-1)  # (seq_len, batch_size, x_dim, K)
    var_x = logvar_x_t.exp() # (seq_len, batch_size, x_dim, K)
    
    rec_loss = torch.div((x_t_extended - mu_x_t)**2, var_x)  # (seq_len, batch_size, x_dim, K) - x_t_extended is broadcasted along last axis
    rec_loss += logvar_x_t # (seq_len, batch_size, x_dim, K)
    
    rec_loss = torch.mean(rec_loss, dim=3)  # Mean over the K samples - (seq_len, batch_size, x_dim)
    rec_loss = torch.sum(rec_loss, dim=2)  # Sum over the x_dim - (seq_len, batch_size)
    rec_loss += x_dim * torch.log(2 * torch.tensor(torch.pi))  # (seq_len, batch_size) - pour normaliser
    rec_loss = torch.sum(rec_loss, dim=0)  # Sum over the sequence length - (batch_size)
    rec_loss = torch.mean(rec_loss)  # Mean over the batch - ()
    
    rec_loss = 1/2 * (rec_loss / seq_len)
    
    # compute the expectation of the KL divergence loss with K samples
    
    kl_loss = logvar_theta_z_t - logvar_phi_z_t  # (seq_len, batch_size, z_dim, K)
    kl_loss += torch.div(logvar_phi_z_t.exp(), logvar_theta_z_t.exp()) # (seq_len, batch_size, z_dim, K)
    kl_loss += torch.div((mu_theta_z_t - mu_phi_z_t).pow(2), logvar_theta_z_t.exp())
       
    kl_loss = torch.mean(kl_loss, dim=3)  # Mean over the K samples - (seq_len, batch_size, z_dim)
    kl_loss = torch.sum(kl_loss, dim=2)  # Sum over the z_dim - (seq_len, batch_size)
    kl_loss -= z_dim # shape (seq_len, batch_size), normalisation
    kl_loss = torch.sum(kl_loss, dim=0)  # Sum over the sequence length - (batch_size)
    kl_loss = torch.mean(kl_loss)  # Mean over the batch
    
    kl_loss = 1/2 * kl_loss / seq_len
    
    return rec_loss, kl_loss, rec_loss + beta * kl_loss



#----------------------------------------------------------------------
#
# TRAIN, TEST steps and TRAINING funtions
#
#----------------------------------------------------------------------

def train_step(model, optimizer, loss_fn, train_loader=None, device=None, beta=None, K=None):
    ### training step
    model.train()
    optimizer.zero_grad()
    ### loop on training data
    rec_loss = 0
    kl_loss = 0
    epoch_loss = 0
    ### check on K
    if K is None:
        K=1
        
    x_dim = model.input_dim
    latent_dim = model.latent_dim
    
    for input, _ in train_loader:
        input = input.to(device).unsqueeze(-1)  # add a feature dimension
        input = input.permute(1, 0, 2)  # permute to (seq_len, batch_size, input_dim)

        
        mu_x_t = torch.zeros(input.shape[0], input.shape[1], x_dim, K).to(device)
        logvar_x_t = torch.zeros(input.shape[0], input.shape[1], x_dim, K).to(device)
        mu_phi_z_t = torch.zeros(input.shape[0], input.shape[1], latent_dim, K).to(device)
        logvar_phi_z_t = torch.zeros(input.shape[0], input.shape[1], latent_dim, K).to(device)
        mu_theta_z_t = torch.zeros(input.shape[0], input.shape[1], latent_dim, K).to(device)
        logvar_theta_z_t = torch.zeros(input.shape[0], input.shape[1], latent_dim, K).to(device)

        # get K samples of the parameters of each distribution
        for k in range(K):
            # get the parameters of the distributions
            _, mu_x_t[:, :, :, k], logvar_x_t[:, :, :, k], mu_phi_z_t[:, :, :, k], logvar_phi_z_t[:, :, :, k], mu_theta_z_t[:, :, :, k], logvar_theta_z_t[:, :, :, k] = model(input)
        
        rec_loss, kl_loss, total_loss = loss_fn(input, mu_x_t, logvar_x_t, mu_phi_z_t, logvar_phi_z_t, mu_theta_z_t, logvar_theta_z_t, beta=beta)
        
        total_loss.backward()
        optimizer.step()
              
        rec_loss += rec_loss.item()
        kl_loss += kl_loss.item()
        epoch_loss += total_loss.item()
        
    epoch_loss /= len(train_loader)
    rec_loss /= len(train_loader)
    kl_loss /= len(train_loader)
    
    return rec_loss, kl_loss, epoch_loss



# Test step : perform validation test for one epoch

def test_step(model, loss_fn, test_loader=None, device=None, beta=None):
    ### test step
    model.eval()
    
    with torch.no_grad():
        ### loop on test data
        rec_loss = 0
        kl_loss = 0
        epoch_loss = 0
        
        for input, _ in test_loader:
            input = input.to(device).unsqueeze(-1)  # add a feature dimension
            input = input.permute(1, 0, 2)  # permute to (seq_len, batch_size, input_dim)

            _, mu_x_s, logvar_x_s, mu_z_s, logvar_z_s, mu_z_transition_s, logvar_z_transition_s = model(input)
            
            rec_loss, kl_loss, total_loss = loss_fn(input, mu_x_s, logvar_x_s, mu_z_s, logvar_z_s, mu_z_transition_s, logvar_z_transition_s, beta=beta)
                
            rec_loss += rec_loss.item()
            kl_loss += kl_loss.item()
            epoch_loss += total_loss.item()
            
        epoch_loss /= len(test_loader)
        rec_loss /= len(test_loader)
        kl_loss /= len(test_loader)
    
    return rec_loss, kl_loss, epoch_loss


# Train function

def train(model, optimizer, loss_fn, num_epochs=100, train_loader=None, test_loader=None, batch_size=None, device=None, beta=None, beta_scheduler=None, display_frequency=10, K=None):
    
    rec_losses = []
    kl_losses = []
    epoch_losses = []

    val_rec_losses = []
    val_kl_losses = []
    val_epoch_losses = []
    
    betas = []
    rec_loss = 1.0
    
    if K is None:
        K = 1
    else:
        K = int(K)
        
    print(f"Starting training...")
    if device.type == 'cuda':
        print('GPU Name:', torch.cuda.get_device_name(0))
        print('Total GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
    print(f"Number of epochs: {num_epochs}")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']:.3e}")
    print(f"Batch size: {batch_size}")
    if beta_scheduler is not None:
        print(f"Beta scheduler: {beta_scheduler}")
    else:
        print(f"Beta scheduler: None, beta = {beta:.3e}")
    print(f"K = {K}")
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(test_loader.dataset)}")
    print(f"Device: {device}")
    print(f"\n")

    for i in range(num_epochs):
        
        # use the beat scheduler if provided
        if beta_scheduler is not None:
            beta = beta_scheduler(i, rec_loss)
        betas.append(beta)
        
        # run the training step
        rec_loss, kl_loss, epoch_loss = train_step(model, optimizer, loss_fn, train_loader=train_loader, device=device, beta=beta, K=K)

        # log results
        rec_losses.append(rec_loss)
        kl_losses.append(kl_loss)
        epoch_losses.append(epoch_loss)
        
        # run test step
        val_rec_loss, val_kl_loss, val_epoch_loss = test_step(model, loss_fn, test_loader=test_loader, device=device, beta=beta)
            
        # log results
        val_rec_losses.append(val_rec_loss)
        val_kl_losses.append(val_kl_loss)
        val_epoch_losses.append(val_epoch_loss)
        
        # Print the losses for this epoch
        if (i+1) % display_frequency == 0:
            print(f"Epoch {i+1:>5}/{num_epochs} with beta = {beta:.2e} - TRAINING : Rec Loss: {rec_loss:.4e}, KL Loss: {kl_loss:.4e}, Total Loss: {epoch_loss:.4e} - TEST : Rec Loss: {val_rec_loss:.4e}, KL Loss: {val_kl_loss:.4e}, Total Loss: {val_epoch_loss:.4e}")
            
    return rec_losses, kl_losses, epoch_losses, val_rec_losses, val_kl_losses, val_epoch_losses, betas        



#----------------------------------------------------------------------
#
# Beta Schedulers
#
#----------------------------------------------------------------------


class BetaLinearScheduler():
    """Scheduler linéaire simple du beta, rapport entre la loss de reconstruction et la loss KL
    """
    
    def __init__(self, beta_start=0.0, beta_end=1.0, epoch_start=0, epoch_end=None, num_epochs=None):
        """Constructeur

        Args:
            beta_start (float, optional): coefficient beta au démarrage. Defaults to 0.0.
            beta_end (float, optional): coefficient beta en fin de schedule. Defaults to 1.0.
            epoch_start (int, optional): epoch de démarrage du schedule. Defaults to 0.
            epoch_end (int, optional): epoch de fin du schedule. Defaults to None.
            num_epochs (int, optional): nombre d'epochs sur lequel se fait l'évolution du beta. Defaults to None.
            
            Si epoch_end est None, alors num_epochs est utlisé pour la fin du schedule.
            Si num_epochs est None, alors epoch_end est utilisé pour la fin du schedule.
            Si num_epochs et epoch_end sont tous les deux None, alors num_epochs est fixé à 100.
            Si epoch_end et num_epochs sont tous les deux fournis, alors num_epochs est utilisé pour la fin du schedule.
        """
        
        # beta values
        self.beta_start = beta_start
        self.beta_end = beta_end
        
        # epoch profile
        self.epoch_start = int(epoch_start)
        if epoch_end is None and num_epochs is None:
            self.num_epochs = 100
            self.epoch_end = self.epoch_start + self.num_epochs
        elif epoch_end is None and num_epochs is not None:
            self.epoch_end = self.epoch_start + num_epochs
            self.num_epochs = num_epochs
        elif epoch_end is not None and num_epochs is None:
            self.epoch_end = epoch_end
            self.num_epochs = self.epoch_end - self.epoch_start
        else:
            self.num_epochs = num_epochs
            self.epoch_end = self.epoch_start + self.num_epochs

        
    def __call__(self, epoch, rec_loss):
        """Calcul du beta à l'epoch donnée

        Args:
            epoch (int): epoch courante

        Returns:
            float: beta
        """
        
        if epoch < self.epoch_start:
            return self.beta_start
        if epoch > self.epoch_end:
            return self.beta_end
        # beta évolue linéairement entre beta_start et beta_end
        # entre epoch_start et epoch_end
        beta = self.beta_start + (self.beta_end - self.beta_start) * ((epoch - self.epoch_start) / (self.epoch_end - self.epoch_start))
        
        return beta
    
    def __repr__(self):
        """String representation of the scheduler
        """
        
        msg = f"BetaLinearScheduler(beta_start={self.beta_start:.3e}, beta_end={self.beta_end:.3e}, epoch_start={self.epoch_start}, epoch_end={self.epoch_end}, num_epochs={self.num_epochs})"
        
        return msg
    
    
    
class BetaThresholdScheduler():
    """Scheduler du beta, rapport entre la loss de reconstruction et la loss KL.
    Reste constant jusqu'à ce que la rec_loss soit inférieure à un seuil donné.
    Ensuite, le beta évolue linéairement entre beta_start et beta_end pendant num_epochs
    """
    
    def __init__(self, rec_loss_threshold=0.0, beta_start=0.0, beta_end=1.0, num_epochs=None):
        """Constructeur

        Args:
            rec_loss_threshold (float, optional): seuil de la loss de reconstruction. Defaults to 0.0.
            beta_start (float, optional): coefficient beta au démarrage. Defaults to 0.0.
            beta_end (float, optional): coefficient beta en fin de schedule. Defaults to 1.0.
            num_epochs (int, optional): nombre d'epochs sur lequel se fait l'évolution du beta. Defaults to None.
        """
        
        # beta values
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule_started = False
        
        # epoch profile
        if num_epochs is None:
            self.num_epochs = 100
        else:
            self.num_epochs = num_epochs
            
        # rec_loss thershold
        self.rec_loss_threshold = rec_loss_threshold

        
    def __call__(self, epoch, rec_loss):
        """Calcul du beta

        Args:
            rec_loss (float): loss de reconstruction

        Returns:
            float: beta
        """
        
        if self.beta_schedule_started is False:
            if rec_loss < self.rec_loss_threshold:
                self.beta_schedule_started = True
                self.epoch = 0
            return self.beta_start
        else:     
            # beta évolue linéairement entre beta_start et beta_end
            # entre epoch_start et num_epochs
            beta = self.beta_start + (self.beta_end - self.beta_start) * (self.epoch / self.num_epochs)
            if self.epoch < self.num_epochs:
                self.epoch += 1
                
            return beta
    
    def __repr__(self):
        """String representation of the scheduler
        """
        
        msg = f"BetaThresholdScheduler(beta_start={self.beta_start:.3e}, beta_end={self.beta_end:.3e}, num_epochs={self.num_epochs}, seuil rec_loss={self.rec_loss_threshold:.3e})"
        
        return msg
    
    
    
#----------------------------------------------------------------------
#
# Plotting functions
#
#----------------------------------------------------------------------



def plot_losses(rec_losses, kl_losses, epoch_losses, val_rec_losses, val_kl_losses, val_epoch_losses, betas):
    
    # Plot the losses

    fig, axs = plt.subplots(1, 4, figsize=(20, 3))

    axs[0].plot(torch.tensor(rec_losses).cpu().detach(), label='Training', color='blue')
    axs[0].plot(torch.tensor(val_rec_losses).cpu().detach(), label='Test', color='green')
    axs[0].set_title('Reconstruction Loss')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid()

    axs[1].plot(torch.tensor(kl_losses).cpu().detach(), label='Training', color='blue')
    axs[1].plot(torch.tensor(val_kl_losses).cpu().detach(), label='Test', color='green')
    axs[1].set_title('KL Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid()

    axs[2].plot(torch.tensor(epoch_losses).cpu().detach(), label='Training', color='blue')
    axs[2].plot(torch.tensor(val_epoch_losses).cpu().detach(), label='Test', color='green')
    axs[2].set_title('Total Loss')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Loss')
    axs[2].legend()
    axs[2].grid()
    
    axs[3].plot(betas, label='Beta', color='blue')
    axs[3].set_title('Beta schedule')
    axs[3].set_xlabel('Epochs')
    axs[3].set_ylabel('Beta')
    axs[3].legend()
    axs[3].grid()

    plt.tight_layout()
    plt.show()
    
    
    
#----------------------------------------------------------------------
#
#  Sampling predictions
#
#----------------------------------------------------------------------


def sample_predictions(N_SAMPLES=3, model=None, X_valid=None, y_valid=None, n_steps=None, n_ahead=None, device=None):
    
    idx = np.random.randint(0, len(X_valid), N_SAMPLES)
    X_valid_subset = X_valid[idx]
    y_valid_subset = y_valid[idx]
    
    fig, axs = plt.subplots(N_SAMPLES, 1, figsize=(16, 3 * N_SAMPLES))
    for i in range(N_SAMPLES):
        input = torch.tensor(X_valid_subset[i], device=device).unsqueeze(1).unsqueeze(2)
        # print(f"input shape : {input.shape}")
        target = torch.tensor(y_valid_subset[i], device=device)
        target = target.cpu().detach().numpy()
        mu_predictions, logvar_predictions, mu_full_x, logvar_full_x = model.predict(input, n_ahead)
              
        # display data
        axs[i].plot(input.squeeze().cpu().detach().numpy(), color='blue', marker=".", linewidth=1, label="input")
        axs[i].plot(np.arange(len(target))+n_steps, target, color='red', marker="o", linewidth=1, label="ground truth")
        
        # display predictions and credible intervals
        all_times = np.arange(n_steps+n_ahead)
        mu_full_x = mu_full_x.squeeze().cpu().detach().numpy()
        logvar_full_x = logvar_full_x.squeeze().cpu().detach().numpy()
        std_full_x = np.exp(logvar_full_x / 2)
        
        axs[i].scatter(all_times, mu_full_x, color='green', marker="x", linewidth=1, label="reconstructed and predicted")
        axs[i].fill_between(all_times, mu_full_x-2*std_full_x, mu_full_x+2*std_full_x, color='orange', label='+/- 2 std', alpha=0.2)
        
        axs[i].set_title(f"Time series {idx[i]}")
        axs[i].set_xlabel("Time")
        axs[i].set_ylabel("Value")
        axs[i].legend()
        axs[i].grid(True)
        
    plt.tight_layout()
    plt.show()