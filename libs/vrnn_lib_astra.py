#------------------------------------------------------------------
#
#
#------   MODELs, FUNCTIONS, CLASSES for Variationnal RNNs   ------
#
#
#------------------------------------------------------------------

#--------------------------------------------------------------------
#
# Default parameters
#
#--------------------------------------------------------------------

X_DIM = 1 # Dimension of the observation space
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

    
#--- brick 3 : Encoder q_phi -----------------------------
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
    

#--- brick 4 : Latent State Transition MLP, p_theta_z -----------------------------

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







   
    
    
