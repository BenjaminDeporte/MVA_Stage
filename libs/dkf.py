#-------------------------------------------------------------------
#
#   18 MAI 2025 : obsolete. replaced by dkf_lib.py
#
#--------------------------------------------------------------------


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

# if torch.cuda.is_available():
#     torch.set_default_device('cuda')
#     device = torch.device('cuda')
# else:
#     torch.set_default_device('cpu')
#     device = torch.device('cpu')

# print(f"Using {device}")

# if device.type == 'cuda':
#     print('GPU Name:', torch.cuda.get_device_name(0))
#     print('Total GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory/1024**3,1), 'GB')
    
# set default tensor type ------------------------------------------

# torch.set_default_dtype(torch.float64)

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
    Creates a unidirectional LSTM, with num_layers layers, hidden state
    of dimension hidden_size, and input of dimension input_size.
    The LSTM is used to process the input sequence in reverse order.
    """
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(BackwardLSTM, self).__init__()
        """
        input_size : dimension of the observation space
        hidden_size : dimension of the hidden state of the LSTM network
        num_layers : number of layers of the LSTM network (default = 1)
        """
        
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
        """
        Forward pass of the backward LSTM module. The sequence x
        is reversed via a torch.flip operation, then passed through the LSTM.
        The output sequence is reversed again to get the final output.
        
        Args:
            x: input sequence
            shape (seq_len, batch, input_size)
        Returns:
            out: output sequence
            shape (seq_len, batch, hidden_size)
        """
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
    and the (sampled) latent variable at time t-1, to compute a tensor g at time t,
    that will be used to compute the parameters of the approximate posterior
    distribution of the latent variable.
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, 
                 hidden_dim=H_DIM, 
                 output_dim=G_DIM,
                 layers_dim = None,  # list of layers dimensions, without the input dimnesion, without the output dimension
                 activation = 'tanh',
                 inter_dim = INTERMEDIATE_LAYER_DIM,
                 ):
        """Creates a combiner module, ie a MLP layer, to combine h_t (the output of the backward lstm at time t
        and (sampled) z_t-1, the latent variable at time t-1.
        The input_dimension is the sum of the latent dimension (dimension of z_t-1) and the hidden dimension
        (dimension of h_t).
        The output dimension is the dimension of the combiner output, which is a parameter.
        The list of intermediate layers is passed in the list layers_dim.
        The activation function is passed as a string, either 'tanh' or 'relu'. (default = 'tanh')
        
        Inputs:
            latent_dim : dimension of the latent space
            hidden_dim : dimension of the hidden state of the LSTM network
            output_dim : dimension of the combiner output
            layers_dim : list of layers dimensions, without the input dimension, without the output dimension
            activation : activation function (default = 'tanh')
            inter_dim : dimension of the intermediate layers (default = INTERMEDIATE_LAYER_DIM)
        """
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
            self.layers_dim = [self.inter_dim]
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
            
        # build the MLP
        self.mlp = nn.Sequential(*layers)
            
        
    def forward(self, h, z):
        """
        Forward pass of the combiner module.
        The use of the combiner module is sequential, so there is no seq_len dimension.
        
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
        x = torch.cat((h, z), dim=-1) # shape (batch, hidden_dim + latent_dim)
        
        # Pass through MLP
        g = self.mlp(x) # output shape (batch, output_dim)
        
        return g     
        
#--- brick 3 : Encoder -----------------------------
#
# This computes the parameters of the approximate posterior distribution
# of the latent vatiable at time t.
# The approximate posterior distribution is a Gaussian distribution,
# we use a MLP to compute the mean and the log of the variance.
#

class EncoderMLP(nn.Module):
    """Encoder module. 
    Computes the parameters of the approximate posterior
    distribution of the latent variable at time t. 
    The approximate posterior distribution is a Gaussian distribution, we use a MLP to compute the mean
    and the log of the variance.
    The input variables are g_t (the output of the combiner module at time t) and
    z_t-1 (the -sampled- latent variable at time t-1).
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, # Dimension of the latent space
                 combiner_dim=G_DIM, # Dimension of the combiner output
                 inter_dim=INTERMEDIATE_LAYER_DIM, # Dimension of the intermediate layers
                 layers_dim = None, # Dimension of the MLP layers (without input nor output)
                 activation = 'tanh', # Activation function
    ):
        """Creates the encoder module.
        This is a MLP, that takes as input the combiner output g_t (combining h_t and z_t-1).
        The input dimension is the combiner dimension.
        The output dimension is the 2 * dimension of the latent space (mean, and log variance).
        The list of intermediate layers is passed in the list layers_dim.
        The activation function is passed as a string, either 'tanh' or 'relu'. (default = 'tanh')
        
        Inputs:
            latent_dim : dimension of the latent space
            combiner_dim : dimension of the combiner output
            inter_dim : dimension of the intermediate layers (default = INTERMEDIATE_LAYER_DIM)
            layers_dim : list of layers dimensions, without the input dimension, without the output dimension
            activation : activation function (default = 'tanh')
        """
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
                layers.append(nn.Linear(self.combiner_dim, dim))
            else:
                layers.append(nn.Linear(layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(self.layers_dim[-1], 2 * latent_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, g):
        """
        Forward pass of the encoder module.
        Takes g_t (the output of the combiner module at time t) as input, returns the parameters of the
        approximate posterior distribution of the latent variable at time t.
        The encoder module is used sequentially, so there is no seq_len dimension.
        The output is a tuple (mu, logvar), where mu is the mean of the approximate posterior distribution
        and logvar is the log of the variance of the approximate posterior distribution.
        
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
    """Latent space transition module. 
    Computes the parameters of the transition distribution of the latent variable at time t (ie "prior").
    The transition distribution is a Gaussian distribution, we use a MLP to compute the mean
    and the log of the variance.
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, # Dimension of the latent space
                 inter_dim=INTERMEDIATE_LAYER_DIM, # Dimension of the intermediate layers
                 layers_dim = None, # Dimension of the MLP layers
                 activation = 'tanh', # Activation function
    ):
        """Creates the latent space transition MLP.
        This is a MLP, that takes as input the -sampled- lagged latent variable z_t-1,
        and computes the parameters of the transition distribution p(z_t|z_{t-1}).
        The input dimension is the latent dimension.
        The output dimension is the 2 * dimension of the latent space (mean, and log variance).
        The list of intermediate layers is passed in the list layers_dim.
        The activation function is passed as a string, either 'tanh' or 'relu'. (default = 'tanh')
        Inputs:
            latent_dim : dimension of the latent space
            inter_dim : dimension of the intermediate layers (default = INTERMEDIATE_LAYER_DIM)
            layers_dim : list of layers dimensions, without the input dimension, without the output dimension
            activation : activation function (default = 'tanh')
        """
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
        self.layers_dim = layers_dim
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(self.layers_dim):
            if i==0:
                layers.append(nn.Linear(self.latent_dim, dim))
            else:
                layers.append(nn.Linear(self.layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(self.layers_dim[-1], 2 * self.latent_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
               
    def forward(self, z):
        """
        Forward pass of the latent space transition module.
        This takes as input a whole set of lagged -sampled- latent variables z_t-1,
        and computes the parameters of the transition distribution p(z_t|z_{t-1}) for each time step t.
        
        Args:
            z: latent variables lagged(set of latent variables at time t-1)
            shape (seq_len, batch, latent_dim)
            
        Returns:
            mu: means of the transition distribution
            shape (seq_len, batch, latent_dim)
            logvar: log of the variances of the transition distribution
            shape (seq_len, batch, latent_dim)
        """
        
        # Pass through MLP
        out = self.mlp(z)
        
        # Split the output into mean and log variance
        # each with shape (batch, latent_dim)
        mu, logvar = out[:, :, :self.latent_dim], out[:, :, self.latent_dim:]
        
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
    observed variable 'x', given the latent variable 'z' sampled from the approximate
    posterior distribution.
    The distribution is a Gaussian distribution, we use a MLP to compute the mean and the log of
    the variance.   
    """
    
    def __init__(self, 
                 latent_dim=Z_DIM, # Dimension of the latent space
                 observation_dim=X_DIM, # Dimension of the observation space
                 inter_dim=INTERMEDIATE_LAYER_DIM, # Dimension of the intermediate layers
                 layers_dim = None, # Dimension of the MLP layers
                 activation = 'tanh', # Activation function
    ):
        """Creates the decoder module.
        This is a MLP, that takes as input a set of -sampled- latent variables z_t,
        and computes the parameters of the distribution of the observed variable 'x'.
        The input dimension is the latent dimension.
        The output dimension is the 2 * dimension of the observation space (mean, and log variance).
        The list of intermediate layers is passed in the list layers_dim.
        The activation function is passed as a string, either 'tanh' or 'relu'. (default = 'tanh')
        
        Inputs:
            latent_dim : dimension of the latent space
            observation_dim : dimension of the observation space
            inter_dim : dimension of the intermediate layers (default = INTERMEDIATE_LAYER_DIM)
            layers_dim : list of layers dimensions, without the input dimension, without the output dimension
            activation : activation function (default = 'tanh')
        """
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
        self.layers_dim = layers_dim
            
        # explicitly define the MLP layers
        layers = []
        for i, dim in enumerate(self.layers_dim):
            if i==0:
                layers.append(nn.Linear(self.latent_dim, dim))
            else:
                layers.append(nn.Linear(self.layers_dim[i-1], dim))
            layers.append(self.activation_fn)
            
        # last layer is linear, no activation
        layers.append(nn.Linear(self.layers_dim[-1], 2 * self.observation_dim)) 
                    
        # build the MLP
        self.mlp = nn.Sequential(*layers)
        
    def forward(self, z):
        """
        Forward pass of the decoder module.
        This takes as input the set of -sampled- latent variables z_t, and 
        computes the set of parameters of the distribution of the observed variable 'x'.
        The decoder module is used after the sequential loop, so there is a seq_len dimension.
        
        Args:
            z: latent variable at time t
            shape (seq_len, batch, latent_dim)
        Returns:
            mu: means of the distribution of the observed variable
            shape (seq_len, batch, observation_dim)
            logvar: log of the variances of the distribution of the observed variable
            shape (seq_len, batch, observation_dim)
        """
        # Pass through MLP
        out = self.mlp(z)
        
        # Split the output into mean and log variance
        # each with shape (batch, observation_dim)
        mu, logvar = out[:, :, :self.observation_dim], out[:, :, self.observation_dim:]
        
        return mu, logvar
    
    
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
    

#-------------------------------------------------------------
#
# class DeepKalmanFilter
#
#--------------------------------------------------------------

class DeepKalmanFilter(nn.Module):
    """
    Deep Kalman Filter (DKF) module. Implements the DKF algorithm.
    
    Args:
        nn (_type_): _description_
        
    Returns:
        _type_: _description_
    """
    
    def __init__(self,
                 input_dim=X_DIM, # Dimension of the observation space
                 latent_dim=Z_DIM, # Dimension of the latent space
                 hidden_dim=H_DIM, # Dimension of the hidden state of the LSTM network
                 combiner_dim=G_DIM, # Dimension of the combiner output
                 inter_dim=INTERMEDIATE_LAYER_DIM, # Dimension of the intermediate layers
                 activation='tanh', # Activation function
                 num_layers=1, # Number of layers of the LSTM network
                 device='cpu' # Device to use (cpu or cuda)
                 ):
        super(DeepKalmanFilter, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.combiner_dim = combiner_dim
        self.inter_dim = inter_dim
        self.device = device
        
        # define the modules
        
        self.backward_lstm = BackwardLSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=num_layers
        )
        
        self.combiner = CombinerMLP(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.combiner_dim,
            activation=activation,
            layers_dim=None, # list of layers dimensions, without the input dimension, without the output dimension
            inter_dim=self.inter_dim
        )
        
        self.encoder = EncoderMLP(
            latent_dim=self.latent_dim,
            combiner_dim=self.combiner_dim,
            inter_dim=self.inter_dim,
            activation=activation,
            layers_dim=None, # list of layers dimensions, without the input dimension, without the output dimension
        )
        
        self.latent_space_transition = LatentSpaceTransitionMLP(
            latent_dim=self.latent_dim,
            inter_dim=self.inter_dim,
            activation=activation,
            layers_dim=None, # list of layers dimensions, without the input dimension, without the output dimension
        )
        
        self.decoder = DecoderMLP(
            latent_dim=self.latent_dim,
            observation_dim=self.input_dim,
            inter_dim=self.inter_dim,
            activation=activation,
            layers_dim=None, # list of layers dimensions, without the input dimension, without the output dimension
        )
        
        self.sampler = Sampler()
        
    def forward(self, x_t):
        """
        Forward pass of the Deep Kalman Filter. 
        
        Runs one step inference :
        
        0- Initialization are run (sampled latent variable)
        1- The input sequence x_t (seq_len, batch, input_dimension) is passed through the backward LSTM 
           to get the hidden states h_t (seq_len, batch, hidden_dim).
        2- A sequential loop is run from time t=1 to t=T (seq_len):
            2.1- the sampled latent variable at previous time step z_t-1, and the hidden state h_t, are run
                 through the combiner module to get the tensor g_t.
            2.2- the tensor g_t is passed through the encoder module to get the parameters of the approximate posterior 
                 distribution mu_phi and logvar_phi of the latent variable z_t.
            2.3- the  latent variable z_t at time t is sampled from the approximate posterior distribution
                 using the reparameterization trick. The sampled latent variable z_t is stored in the sequence of
                 sampled latent variables z_t_s.
        3- The whole sequence of sampled latent variables z_t_s is passed through the decoder module
           to get the parameters of the distribution of the observed variables x_t.
        4- The whole sequence of sampled latent variables z_t_s, lagged by one time step, is passed through the
           transition module to get the parameters of the transition distribution of the latent variable z_t. 
        
        Args:
            x_t: input sequence - shape (seq_len, batch, input_dim)
            
        Intermediate variables:
            sampled_z_t : sequence of sampled latent variables - shape (seq_len, batch, latent_dim).
            NB : z_0 is set to 0.
            h_t : hidden state of the backward LSTM at time t - shape (seq_len, batch, hidden_dim)
            
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
        sampled_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        z0 = torch.zeros(batch_size, self.latent_dim).to(self.device)  # initial latent variable z_0
        mu_phi_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        logvar_phi_z_t = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
        
        # step 1 : run the backward LSTM on the input sequence
        # outputs are the hidden states, shape (seq_len, batch, hidden_dim)
        h_t = self.backward_lstm(x_t)
        
        # step 2 : loop from t=1 to t=T (seq_len) to compute and sampled the latent variables z_t
        for t in range(seq_len):
            
            # step 2.1 : at time t, get the sampled latent variable z_t-1 and the hidden state h_t
            if t == 0:
                sampled_z_t_1 = z0
            else:
                sampled_z_t_1 = sampled_z_t[t-1]
            # combine them to compute g_t
            g_t = self.combiner(h_t[t], sampled_z_t_1) # shpae is (batch_size, combiner_dim)
            
            # step 2.2 : compute the parameters of the approximate posterior distribution
            mu_phi, logvar_phi = self.encoder(g_t) 
            mu_phi_z_t[t], logvar_phi_z_t[t] = mu_phi, logvar_phi
            
            # step 2.3 : sample z_t from the approximate posterior distribution and store it
            sampled_z_t[t] = self.sampler(mu_phi, logvar_phi)
            
        # step 3 : compute the parameters of the observation distribution
        mu_x_t, logvar_x_t = self.decoder(sampled_z_t)
        
        # step 4 : compute the parameters of the transition distribution
        # form the lagged sampled latent variable z_t : z_t[0:seq_len-1]
        lagged_sampled_z_t = torch.cat([z0.unsqueeze(0), sampled_z_t[:-1]])
        mu_theta_z_t, logvar_theta_z_t = self.latent_space_transition(lagged_sampled_z_t) 
                        
        # return the outputs
        return x_t, mu_x_t, logvar_x_t, mu_phi_z_t, logvar_phi_z_t, mu_theta_z_t, logvar_theta_z_t
    
    def __repr__(self):
        
        msg = f"DeepKalmanFilter(input_dim={self.input_dim}, latent_dim={self.latent_dim}, hidden_dim={self.hidden_dim}, combiner_dim={self.combiner_dim}, inter_dim={self.inter_dim})"
        msg += f"\n{self.backward_lstm}"
        msg += f"\n{self.combiner}"
        msg += f"\n{self.encoder}"
        msg += f"\n{self.latent_space_transition}"
        msg += f"\n{self.decoder}"
        msg += f"\n{self.sampler}"
        
        return msg
    
    def predict(self, x, num_steps):
        """
        Predicts future steps based on the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (seq_len, batch_size, x_dim).
            num_steps (int): Number of future steps to predict.

        Returns:
            predictions (torch.Tensor): Tensor of shape (num_steps, batch_size, x_dim)
            containing the predicted future steps.
            full_x (torch.Tensor): Tensor of shape (seq_len + num_steps, batch_size, x_dim)
            containing the reconstrcuted input sequence and the predicted future steps.
            
            
        """
        
        with torch.no_grad():
            
            # run an inference forward pass to learn the latent variable transition probability distrubution
            seq_len, batch_size, input_dim = x.shape
            assert input_dim == self.input_dim, f"Input dimension {input_dim} does not match the expected dimension {self.input_dim}"
            
            z0 = torch.zeros(batch_size, self.latent_dim).to(self.device)
            # initialize the hidden state of the backward LSTM at time t=0
            # NB : they are not used in a first version of this code
            h0 = torch.zeros(batch_size, self.hidden_dim).to(self.device)
            c0 = torch.zeros(batch_size, self.hidden_dim).to(self.device)
            
            # initialize the outputs
            mu_z_s, logvar_z_s = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device), torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
            
            # run the backward LSTM on the input sequence
            # outputs are the hidden states, shape (seq_len, batch, hidden_dim)
            h_t_s = self.backward_lstm(x)
            
            # loop to compute the approximate posterior distribution of the latent variables z_t
            # given the observations x_t
            # initialize the sequence of sampled latent variables z_t
            sampled_z_t_s = torch.zeros(seq_len, batch_size, self.latent_dim).to(self.device)
            
            for t in range(seq_len):
                # at time t, get z_t-1 and h_t
                if t == 0:
                    sampled_z_t_1 = z0
                else:
                    sampled_z_t_1 = sampled_z_t_s[t-1]
                h_t = h_t_s[t]
                # compute g_t
                g_t = self.combiner(h_t, sampled_z_t_1)
                # compute the parameters of the approximate posterior distribution
                mu_z, logvar_z = self.encoder(g_t)
                mu_z_s[t], logvar_z_s[t] = mu_z, logvar_z
                # sample z_t and store it
                sampled_z_t = self.sampler(mu_z, logvar_z)
                sampled_z_t_s[t] = sampled_z_t
                
            # compute the parameters of the transition distribution
            z_t_lagged = torch.cat([z0.unsqueeze(0), sampled_z_t_s[:-1]])  # lagged z_t
            mu_z_transition_s, logvar_z_transition_s = self.latent_space_transition(z_t_lagged)
            
            # compute the parameters of the observation distribution
            mu_x_s, logvar_x_s = self.decoder(sampled_z_t_s)
            
            # use mu_x as the reconstructed x
            x_hat = mu_x_s
            
            # Start with the last inferred latent state
            z_pred = sampled_z_t_s[-1:, :, :]

            # Start to predict x at the end of the given sequence
            predictions = torch.zeros(num_steps, batch_size, self.input_dim).to(self.device)
            for s in range(num_steps):
                # Get the parameters (mean and variance) of the transition
                # distribution p(z_t|z_{t-1})
                z_pred_mean, z_pred_logvar = self.latent_space_transition(z_pred)

                # Sample from p(z_t|z_{t-1}) distribution
                z_pred = self.sampler(z_pred_mean, z_pred_logvar)
                
                x_pred, _ = self.decoder(z_pred)
                
                predictions[s,:,:] = x_pred

        # Append predictions to the reconstructed x
        full_x = torch.cat([x_hat, predictions], dim=0)
        
        return predictions, full_x
    

# VERSION PRECEDENTE DE LA LOSS  
# def loss_function(x, x_hat, x_hat_logvar, z_mean, z_logvar,
#                   z_transition_mean, z_transition_logvar, beta=1.0):
#     """
#     Compute the total loss for a variational autoencoder (VAE) with a weighted 
#     reconstruction loss and a Kullback-Leibler (KL) divergence term.

#     Parameters:
#     -----------
#     x : torch.Tensor
#         Ground truth data with shape (seq_len, batch_size, x_dim).
#     x_hat : torch.Tensor
#         Reconstructed data from the VAE with shape
#         (seq_len, batch_size, x_dim).
#     x_hat_logvar : torch.Tensor
#         Log variance of the reconstructed data with shape
#         (seq_len, batch_size, x_dim).
#     z_mean : torch.Tensor
#         Mean of the latent variable distribution with shape 
#         (seq_len, batch_size, x_dim).
#     z_logvar : torch.Tensor
#         Log variance of the latent variable distribution with shape 
#         (seq_len, batch_size, x_dim).
#     z_transition_mean : torch.Tensor
#         Mean of the transition distribution in the latent space with shape 
#         (seq_len, batch_size, x_dim).
#     z_transition_logvar : torch.Tensor
#         Log variance of the transition distribution in the latent space with 
#         shape (seq_len, batch_size, x_dim).
#     beta : float
#         Weighting factor for the KL divergence term.
#     loss_type : str
#         Type of reconstruction loss to use. Options:
#         - 'mse': Mean Squared Error (MSE) loss.
#         - 'weighted_mse': Weighted Mean Squared Error (MSE) loss.

#     Returns:
#     --------
#     total_loss : torch.Tensor
#         The total loss, which is the sum of the reconstruction loss and the 
#         KL divergence loss.

#     Notes:
#     ------
#     - The "reconstruction loss" is based on formula above
#     - The KL divergence loss measures the difference between the latent
#       variable distribution and the transition distribution in the latent space.
#     - Both losses are normalized by the sequence length (`seq_len`) and
#       averaged over the batch.
#     - The total loss is a combination of the reconstruction loss and the 
#       KL divergence loss, weighted by the `beta` parameter.
#     """
    
#     seq_len, batch_size, x_dim = x.shape
    
#     # Compute the reconstruction loss
#     var = x_hat_logvar.exp()
#     loss = torch.div((x - x_hat)**2, var)
            
#     loss += x_hat_logvar
#     loss = loss.sum(dim=2)  # Sum over the x_dim
#     loss = loss.sum(dim=0)  # Sum over the sequence length
#     loss = loss.mean()  # Mean over the batch
#     reconstruction_loss = loss / seq_len
           
#     # Compute the KL divergence loss
#     kl_loss = (z_transition_logvar - z_logvar +
#                torch.div((z_logvar.exp() + 
#                          (z_transition_mean - z_mean).pow(2)),
#                          z_transition_logvar.exp()))
    
#     kl_loss = kl_loss.sum(dim=2)  # Sum over the z_dim
#     kl_loss = kl_loss.sum(dim=0)  # Sum over the sequence length
#     kl_loss = kl_loss.mean()  # Mean over the batch
#     kl_loss = kl_loss / seq_len
                
#     # Combine the reconstruction loss and the KL divergence loss
#     total_loss = reconstruction_loss + beta * kl_loss
    
#     return reconstruction_loss, kl_loss, total_loss

def loss_function(x_t, mu_x_t, logvar_x_t, mu_phi_z_t, logvar_phi_z_t, mu_theta_z_t, logvar_theta_z_t, beta=None):
    """
    Version 2.0 of the loss function for the Deep Kalman Filter.
    All distributions parameters are assumed to have been computed from K samples of each z_t.
    The K samples are used to compute the expectations within the loss function.

    Args:
        x_t (tensor): the original input sequence - shape (seq_len, batch_size, x_dim)

        mu_x_t (tensor): the mean of the distribution of the observed variable - shape (seq_len, batch_size, x_dim, K)
        logvar_x_t (tensor): the log of the variance of the distribution of the observed variable - shape (seq_len, batch_size, x_dim, K)
        mu_phi_z_t (tensor): the mean of the approximate posterior distribution (q_\phi) of the latent variable - shape (seq_len, batch_size, x_dim, K)
        logvar_phi_z_t (tensor): the log of the variance of the approximate posterior distribution (q_\phi) of the latent variable - shape (seq_len, batch_size, x_dim, K)
        mu_theta_z_t (tensor): the mean of the transition distribution (p_\theta_z) of the latent variable - shape (seq_len, batch_size, x_dim, K)
        logvar_theta_z_t (tensor): the log of the variance of the transition distribution (p_\theta_z) of the latent variable - shape (seq_len, batch_size, x_dim, K)

        beta (float, optional): the weight of the KL divergence term in the loss function. Defaults to None.
        
    Returns:
        rec_loss (tensor): the reconstruction loss - shape (1,)
        kl_loss (tensor): the KL divergence loss - shape (1,)
        total_loss (tensor): the total loss - shape (1,)
    """
    
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
    rec_loss = torch.sum(rec_loss, dim=0)  # Sum over the sequence length - (batch_size)
    rec_loss = torch.mean(rec_loss)  # Mean over the batch - ()
    
    rec_loss = 1/2 * (rec_loss / seq_len)
    
    # compute the expectation of the KL divergence loss with K samples
    
    kl_loss = logvar_theta_z_t - logvar_phi_z_t  # (seq_len, batch_size, z_dim, K)
    kl_loss += torch.div(logvar_phi_z_t.exp(), logvar_theta_z_t.exp()) # (seq_len, batch_size, z_dim, K)
    kl_loss += torch.div((mu_theta_z_t - mu_phi_z_t).pow(2), logvar_theta_z_t.exp())
    # kl_loss -= z_dim # shape (seq_len, batch_size, z_dim, K)
       
    kl_loss = torch.mean(kl_loss, dim=3)  # Mean over the K samples - (seq_len, batch_size, z_dim)
    kl_loss = torch.sum(kl_loss, dim=2)  # Sum over the z_dim - (seq_len, batch_size)
    kl_loss = torch.sum(kl_loss, dim=0)  # Sum over the sequence length - (batch_size)
    kl_loss = torch.mean(kl_loss)  # Mean over the batch
    
    kl_loss = 1/2 * kl_loss / seq_len
    
    return rec_loss, kl_loss, rec_loss + beta * kl_loss
    
