#------------------------------------------------------------------
#
#
#------   MODELs, FUNCTIONS, CLASSES for GP-VAE              ------
#
#
#------------------------------------------------------------------

#--------------------------------------------------------------------
#
# Default parameters
#
#--------------------------------------------------------------------

X_DIM = 1 # Dimension of the observation space
Z_DIM = 1 # Dimension of the latent space (first version)
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
# Utilities
#
#--------------------------------------------------------------------

def make_mlp(input_dim, output_dim, n_layers=3, inter_dim=128, activation=nn.ReLU):
    """Create a Multi-Layer Perceptron (MLP) with specified parameters.
    
    Args:
        input_dim (int): Dimension of the input layer.
        output_dim (int): Dimension of the output layer.
        n_layers (int, optional): Number of layers in the MLP. Defaults to 3.
            - first layer is input_dim x inter_dim
            - last layer is inter_dim x output_dim
            - all other layers are inter_dim x inter_dim
        inter_dim (int, optional): Number of neurons in the hidden layers. Defaults to 128.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.ReLU.
            the last layer does not have an activation function.
        
    Returns:
        nn.Sequential: A sequential model representing the MLP.
    """
    
    layers = []
    for i in range(n_layers):
        if i == 0:
            layers.append(nn.Linear(input_dim, inter_dim))
        elif i == n_layers - 1:
            layers.append(nn.Linear(inter_dim, output_dim))
        else:
            layers.append(nn.Linear(inter_dim, inter_dim))
        
        if i < n_layers - 1:
            layers.append(activation())
    
    return nn.Sequential(*layers)
    
    
    
#--------------------------------------------------------------------
#
# CLASS GP-VAE
#
#--------------------------------------------------------------------

#---------------------------------------------------------------------
# ENCODER
#---------------------------------------------------------------------

# brick 0.1 : Encoder Mean

class EncoderMean(nn.Module):
    """ Neural Net to compute the mean of the Gaussian Encoder distribution.
    """
    
    def __init__(self,
                 sequence_length = None,
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 ):
        """_summary_

        Args:
            sequence_length (_type_, optional): Longueur des time series. Doit être spécifié. Defaults to None.
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(EncoderMean, self).__init__()
        
        assert sequence_length is not None, "sequence_length must be specified"
        assert n_layers > 0, "n_layers must be greater than 0"
        assert z_dimension == 1, "the code is not ready for z_dimension > 1 yet, sorry"
        
        self.sequence_length = int(sequence_length)
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
                
        self.mlp = make_mlp(
            input_dim=sequence_length * x_dimension,
            output_dim=sequence_length * z_dimension,  # output is a vector of length sequence_length * z_dimension
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
    
    def forward(self, x):
        """
        Forward pass of the EncoderMean.
        Args:
            x (torch.Tensor): Input tensor (batch_size, sequence_length, x_dimension)
        Returns:
            z (torch.Tensor): Output tensor (batch_size, sequence_length, z_dimension)
                
        NB : z_dimension = 1 in this first implementation.
        """
        
        assert x.dim() == 3, "Input tensor must have shape (batch_size, sequence_length, x_dimension)"
        
        batch_size = x.size(0)
        sequence_length = x.size(1)
        x_dim = x.size(2)
        
        input = x.view(batch_size, -1)  # Flatten the input tensor (B, L * Dx)
        output = self.mlp(input)  # Apply the MLP, # output has shape (B, L * Dz)
        
        z = output.view(batch_size, sequence_length, self.z_dimension)  # Reshape to (B, L, Dz)
            
        return z
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(sequence_length={self.sequence_length}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__})")
        
        
# brick 0.2 : Encoder CoVariance

class EncoderCovariance(nn.Module):
    """ Neural Net to compute the covariance of the Gaussian Encoder distribution.
    Uses a MLP to compute the lower triangular matrix of the Cholesky decomposition of the covariance matrix.
    L is triangular inferior, with diagonal elements strictly positive.
    """
    
    def __init__(self,
                 sequence_length = None,
                 x_dimension = 1,
                 z_dimension = 1,  # we assume z_dimension = 1 for now
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 ):
        """_summary_

        Args:
            sequence_length (_type_, optional): Longueur des time series. Doit être spécifié. Defaults to None.
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(EncoderCovariance, self).__init__()
        
        assert sequence_length is not None, "sequence_length must be specified"
        assert n_layers > 0, "n_layers must be greater than 0"
        assert z_dimension == 1, "the code is not ready for z_dimension > 1 yet, sorry"
        
        self.sequence_length = int(sequence_length)
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
        
        # this network to get the diagonal elements of the covariance matrix
        self.diagonal_mlp = make_mlp(
            input_dim=sequence_length * x_dimension,
            output_dim=sequence_length,  # output is a vector of length sequence_length (* z_dim = 1)
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
        
        # this network to get a full matrix, of which we well keep only the lower triangular part                
        self.full_matrix_mlp = make_mlp(
            input_dim=self.sequence_length * self.x_dimension,
            output_dim=self.sequence_length * self.sequence_length,  # output is a full matrix of shape (T*T)
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
        
    
    def forward(self, x):
        """
        Forward pass of the EncoderCovariance.
        Args:
            x (torch.Tensor): Input tensor (batch_size, sequence_length, x_dimension) 

        Returns:
            L (torch.Tensor): Lower triangular matrix of shape (batch_size, sequence_length * z_dimension, sequence_length * z_dimension) 
                Here : z_dimension = 1.
            C (torch.Tensor): Covariance matrix of shape (batch_size, sequence_length * z_dimension, sequence_length * z_dimension) 
                Here : z_dimension = 1.
            C is computed as L @ L^T, where L is the lower triangular matrix. (Cholesky decomposition)
        """
        
        # manages shape of input x
        assert x.dim() == 3, "Input tensor must have shape (batch_size, sequence_length, x_dimension)"
        assert x.size(1) == self.sequence_length, f"Wrong sequence length in EncoderCovariance. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        assert x.size(2) == self.x_dimension, f"Wrong x_dim in EncoderCovariance. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        
        input = x.view(x.size(0), -1)  # Flatten the input tensor => x has shape B x (L * Dx)
        
        # Compute the diagonal elements of the covariance matrix
        D = self.diagonal_mlp(input)  # out : (B, T) (Dz=1)
        D = torch.exp(self.diagonal_mlp(input)) # out : (B, T). Ensure strictly positive diagonal elements.
        D = torch.diag_embed(D) # shape (B, T, T)
        
        # Get the elements outside the diagonal
        M = self.full_matrix_mlp(input) # shape (B, T*T)
        M = M.reshape(x.size(0), self.sequence_length * self.z_dimension, self.sequence_length * self.z_dimension) # M is a full matrix of shape (B, T, T)
        idx = torch.arange(self.sequence_length * self.z_dimension, device=x.device)  # Create an index tensor for the diagonal elements
        M[:,idx,idx] = 0.0  # Fill the diagonal with 0s
        
        # Assemble the lower triangular matrix L
        L = torch.zeros_like(M, device=x.device)  # Initialize L with zeros
        L = D + M
        L[:] = torch.tril(L[:])
            
        return L, L @ L.transpose(-1, -2) # (B, T, T) Covariance matrix C
    
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(sequence_length={self.sequence_length}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__})")


# brick 1 : Encoder q_phi

class Encoder(nn.Module):
    """Encoder module. 
    - Takes a sequence of length T of observations x_{1:T}
    - Computes the parameters of the multivariate normal distribution q_phi(z_{1:T}|x_{1:T}):
        - mu_phi(x_{1:T}): mean of the approximate posterior distribution
        - Sigma_phi(x_{1:T}): covariance matrix the approximate posterior distribution
    - Outputs the parameters and the distribution itself.
    """
    
    def __init__(self,                  
                 sequence_length = None,
                 x_dimension = 1,
                 z_dimension = 1,  # we assume z_dimension = 1 for now
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,):
        
        super(Encoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension
        
        self.encoder_mean = EncoderMean(
            sequence_length=sequence_length,
            x_dimension=x_dimension,
            z_dimension=z_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation,
        )
        
        self.encoder_covariance = EncoderCovariance(
            sequence_length=sequence_length,
            x_dimension=x_dimension,
            z_dimension=z_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation,
        )
    
    def forward(self, x):
        """
        Forward pass of the Encoder.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, x_dimension)
            
        Returns:
            mu (torch.Tensor): Mean of the approximate posterior distribution of shape (batch_size, sequence_length)
            Sigma (torch.Tensor): Covariance matrix of the approximate posterior distribution of shape (batch_size, sequence_length, sequence_length)
            q_phi (torch.distributions.MultivariateNormal): Multivariate normal distribution with parameters mu and Sigma.
        """
        
        # manage shape of x
        assert x.dim() == 3, f"Incorrect tensor shape in Encoder. Input tensor must have shape (batch_size, sequence_length, x_dimension). Got {x.shape} instead."
        assert x.size(1) == self.sequence_length, f"Incorrect sequence length passed to Encoder. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        assert x.size(-1) == self.x_dimension, f"Incorrect x_dimension passed to Encoder. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        
        # compute parameters of the approximate posterior distribution
        mu = self.encoder_mean(x)  # (B, L, Dz=1)
        mu = mu.squeeze(-1)  # (B, L)
        _, sigma = self.encoder_covariance(x)  # (B, L, L)
        
        # instantiate the multivariate normal distribution
        q_phi = torch.distributions.MultivariateNormal(mu, sigma)
        
        return mu, sigma, q_phi        
    
    def __repr__(self):
        msg = f"{self.__class__.__name__}(sequence_length={self.sequence_length}, " +\
            f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, " +\
            f"n_layers={self.encoder_mean.n_layers}, inter_dim={self.encoder_mean.inter_dim}, " +\
            f"activation={self.encoder_mean.activation.__name__})" 
        msg += f"\nEncoderMean: {self.encoder_mean}"
        msg += f"\nEncoderCovariance: {self.encoder_covariance}"
        return msg


#--------------------------------------------------------------------------
# DECODER
#--------------------------------------------------------------------------

# brick 0.3 :Decoder Mean

class DecoderMean(nn.Module):
    """ Neural Net to compute the mean of the Gaussian Decoder distribution.
    """
    
    def __init__(self,
                 sequence_length = None,
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 ):
        """_summary_

        Args:
            sequence_length (_type_, optional): Longueur des time series. Doit être spécifié. Defaults to None.
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(DecoderMean, self).__init__()
        
        assert sequence_length is not None, "sequence_length must be specified"
        assert n_layers > 0, "n_layers must be greater than 0"
        assert z_dimension == 1, "the code is not ready for z_dimension > 1 yet, sorry"
        
        self.sequence_length = int(sequence_length)
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
        
        self.mlp = make_mlp(
            input_dim=sequence_length * z_dimension,
            output_dim=sequence_length * x_dimension,  # output is a vector of length sequence_length * x_dimension
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
    
    def forward(self, z):
        """
        Forward pass of the DecoderMean.
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, sequence_length, z_dimension = 1)
        Returns:
            torch.Tensor: mu_x : Output tensor of shape (batch_size, sequence_length, x_dimension)
        """
              
        # mini check
        assert z.dim() == 3, "Incorrect tensor shape in Decoder. Input tensor z must have shape (batch_size, sequence_length, z_dimension)"
        assert z.size(-1) == 1, f"Code not ready for z_dimension > 1 yet, sorry. Input tensor z must have shape (batch_size, sequence_length, {self.z_dimension}) or ({self.z_dimension},) or ({self.z_dimension}, sequence_length) or (sequence_length, {self.z_dimension})"
        assert z.size(-1) == self.z_dimension, f"Incorrect latent dim. Input tensor z must have shape (batch_size, sequence_length, {self.z_dimension}) or ({self.z_dimension},) or ({self.z_dimension}, sequence_length) or (sequence_length, {self.z_dimension})"
        
        # compute stuff
        input = z.view(z.size(0), -1)  # Flatten the input tensor (B, L * Dz)
        output = self.mlp(input)  # Apply the MLP, (B, L * Dx)
        mu_x = output.view(z.size(0), self.sequence_length, self.x_dimension)  # Reshape to (B, L, Dx)
            
        return mu_x
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(sequence_length={self.sequence_length}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__})")
        
# brick 0.4 : Decoder Covariance

class DecoderCovariance(nn.Module):
    """ Neural Net to compute the covariance of the Gaussian Decoder distribution.
    """
    
    def __init__(self,
                 sequence_length = None,
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 ):
        """_summary_

        Args:
            sequence_length (_type_, optional): Longueur des time series. Doit être spécifié. Defaults to None.
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(DecoderCovariance, self).__init__()
        
        assert sequence_length is not None, "sequence_length must be specified"
        assert n_layers > 0, "n_layers must be greater than 0"
        assert z_dimension == 1, "the code is not ready for z_dimension > 1 yet, sorry"
        
        self.sequence_length = int(sequence_length)
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
        
        self.mlp = make_mlp(
            input_dim=sequence_length * z_dimension,
            output_dim=sequence_length * x_dimension,  # output is a vector of length sequence_length
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
    
    def forward(self, z):
        """
        Forward pass of the DecoderCovariance.
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, sequence_length, z_dimension)
        Returns:
            torch.Tensor: Logvar - Output tensor of shape (batch_size, sequence_length, x_dimension)
        """
        
        # mini check
        assert z.dim() == 3, "Incorrect tensor shape in DecoderCovariance. Input tensor z must have shape (batch_size, sequence_length, z_dimension)"
        assert z.size(-1) == 1, f"Code not ready for z_dimension > 1 yet, sorry. Input tensor z must have shape (batch_size, sequence_length, {self.z_dimension}) or ({self.z_dimension},) or ({self.z_dimension}, sequence_length) or (sequence_length, {self.z_dimension})"
        assert z.size(-1) == self.z_dimension, f"Incorrect latent dim. Input tensor z must have shape (batch_size, sequence_length, {self.z_dimension}) or ({self.z_dimension},) or ({self.z_dimension}, sequence_length) or (sequence_length, {self.z_dimension})"
                   
        # compute stuff
        input = z.view(z.size(0), -1)  # Flatten the input tensor (B, L * Dz)
        output = self.mlp(input)  # Apply the MLP, (B, L * Dx)
        logvar_x = output.view(z.size(0), self.sequence_length, self.x_dimension) # Reshape to (B, L, Dx)
            
        return logvar_x
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(sequence_length={self.sequence_length}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__})")
 
# brick 2 : decoder p_\{theta_x}

class GaussianDecoder(nn.Module):
    """Decoder module. Takes a sequence of latent variables and computes the parameters of the
    obervation model p_{\theta_x}. Here, the observation model is a Gaussian distribution,
    and an observation x_t depends only on the latent variable z_t at time t. Therefore,
    the computed covariance matrix is diagonal so the observations are independent.
    
    NB : we also assume that the covariance matrix for N(x_t | mu_x(z_t), sigma_x(z_t)) is diagonal.
    """

    def __init__(self,
                 sequence_length = None,
                 x_dimension = 1,
                 z_dimension = 1,  # we assume z_dimension = 1 for now
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU
                 ):
        
        super(GaussianDecoder, self).__init__()
        
        self.sequence_length = sequence_length
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension
        
        self.decoder_mean = DecoderMean(
            sequence_length=sequence_length,
            x_dimension=x_dimension,
            z_dimension=z_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation,
        )
        
        self.decoder_covariance = DecoderCovariance(
            sequence_length=sequence_length,
            x_dimension=x_dimension,
            z_dimension=z_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation,
        )
    
    def forward(self, z):
        """Takes a sequence of length N of latent variables z_{1:N} (B, N, Dz=1)
        
        Computes the parameters of the observation model p_{\theta_x}(x_{1:N}|z_{1:N}):
            - mu_x(z_{1:N}): mean of the observation model (B, N, Dx)
            - logvar_x(z_{1:N}): log variance of the observation model (B, N, Dx) : this is the diagonal of the covariance matrix
            - p_theta_x: the observation model itself.
        NB : N can be 1, T or another strictly positive integer.
        """
        
        # manage shape of z
        assert z.dim() == 3, "Incorrect z tensor shape passed to Decoder. Input tensor z must have shape (batch_size, sequence_length, z_dimension)"
        assert z.size(-1) == 1, f"Code not ready for z_dimension > 1 yet, sorry. Input tensor z must have shape (batch_size, sequence_length, {self.z_dimension}) or ({self.z_dimension},) or ({self.z_dimension}, sequence_length) or (sequence_length, {self.z_dimension})"
        
        # compute stuff
        mu_x = self.decoder_mean(z)  # (B, N, Dx)
        logvar_x = self.decoder_covariance(z)  # (B, N, Dx)

        # instantiate the multivariate normal distribution
        p_theta_x = torch.distributions.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(logvar_x))) # batch_shape = B, event_shape = (N, Dx)
        
        return mu_x, logvar_x, p_theta_x
    
    def __repr__(self):
        msg = (f"{self.__class__.__name__}(sequence_length={self.sequence_length}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.decoder_mean.n_layers}, inter_dim={self.decoder_mean.inter_dim}, "
                f"activation={self.decoder_mean.activation.__name__})")
        msg += f"\nDecoderMean: {self.decoder_mean}"
        msg += f"\nDecoderCovariance: {self.decoder_covariance}"
        return msg



#---------------------------------------------------------------------
# GAUSSIAN PROCESS PRIOR
#---------------------------------------------------------------------

# --- Mean function -----------------------------------------------

class GPNullMean(nn.Module):
    """ Neural Net to compute the mean of the Gaussian Process prior.
    This is a null mean function, i.e. it returns a tensor of zeros.
    """
    
    def __init__(self,
                 z_dimension = 1,  # z dimension is 1
                 ):
        super(GPNullMean, self).__init__()
        
        # assert z_dimension == 1, "in GPNullMean, code is not ready for z_dimension > 1 yet, sorry"
        self.z_dimension = int(z_dimension)
    
    def forward(self, t):
        """Forward pass of the GPNullMean.
        Just returns a tensor of zeros of shape (batch_size, sequence_length, z_dimension).
        
        Inputs:
            t (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 1)
        Returns:
            torch.Tensor: Output tensor of ZEROS of shape (batch_size, sequence_length, z_dimension)
        """
        assert t.dim() == 3, "In GPMeanNull, Input tensor must have shape (batch_size, sequence_length, x_dimension)"
        assert t.size(-1) == 1, "In GPMeanNull, Input tensor must have shape (batch_size, sequence_length, 1) - this is a set of times"
        
        return torch.zeros_like(t, device=t.device, dtype=t.dtype).repeat(1,1,self.z_dimension)  # returns a tensor of zeros of shape (batch_size, sequence_length, z_dimension)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(z_dimension={self.z_dimension})"
    

# ----- Cauchy Kernel ---------------------------------------------
#
# HIGHLY UNSTABLE KERNEL !  Produces not definite positive matrices from time to time !
#

class CauchyKernel(nn.Module):
    """Cauchy kernel for Gaussian Process.
    NB : requires a small positive constant alpha to ensure positive definiteness of the kernel matrix.
    The lengthscale and variance parameters are learnable (nn.Parameter).
    """
    
    def __init__(self):
        super(CauchyKernel, self).__init__()
        
        # the parameters of the kernel are learnable
        self.lengthscale = nn.Parameter(torch.tensor(1.0))  # learnable lengthscale parameter       
        self.variance = nn.Parameter(torch.tensor(1.0))  # learnable variance parameter
        
        self.alpha = torch.tensor(1.0e-2)  # small positive constant to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Cauchy kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (batch_size, sequence_length, 1)
            t2 (torch.Tensor): Second set of time points (batch_size, sequence_length, 1)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (batch_size, sequence_length, sequence_length)
        """
        
        assert t1.dim() == 3 and t2.dim() == 3, "In kernel computation, Input tensors must have shape (batch_size, sequence_length, 1)"
        assert t1.size(-1) == 1 and t2.size(-1) == 1, "In kernel computation, Input tensors must have last dimension = 1 (this is a set of times)"
        
        diff = t1 - t2.transpose(1, 2) # (B, N, 1) - (B, 1, N) => (B, N, N)
        cauchy_kernel_matrix = torch.divide( self.variance**2, 1.0 + (diff / self.lengthscale)**2  )  # (B, N, N) kernel matrix
        cauchy_kernel_matrix += self.alpha * torch.eye(t1.size(1), device=t1.device, dtype=t1.dtype).unsqueeze(0)  # add identity matrix to ensure positive definiteness
        
        return cauchy_kernel_matrix
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(lengthscale={self.lengthscale.item()}, "
                f"variance={self.variance.item()})")
        

#----- Gaussian Kernel --------------------------------------------
#

class GaussianKernel(nn.Module):  
    """Gaussian kernel for Gaussian Process.
    The lengthscale and variance parameters are learnable (nn.Parameter).
    """
    
    def __init__(self):
        super(GaussianKernel, self).__init__()
        
        # the parameters of the kernel are learnable
        self.lengthscale = nn.Parameter(torch.tensor(1.0))  # learnable lengthscale parameter       
        self.variance = nn.Parameter(torch.tensor(1.0))  # learnable variance parameter
        
        self.alpha = torch.tensor(1.0e-6)  # small positive constant to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Gaussian kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (batch_size, sequence_length, 1)
            t2 (torch.Tensor): Second set of time points (batch_size, sequence_length, 1)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (batch_size, sequence_length, sequence_length)
        """
        
        assert t1.dim() == 3 and t2.dim() == 3, "In kernel computation, Input tensors must have shape (batch_size, sequence_length, 1)"
        assert t1.size(-1) == 1 and t2.size(-1) == 1, "In kernel computation, Input tensors must have last dimension = 1 (this is a set of times)"
        
        kernel = torch.exp(-0.5 * ((t1 - t2.transpose(1, 2)) / self.lengthscale)**2)  # (B, N, N)
        gaussian_kernel_matrix = self.variance * kernel  # (B, N, N)
        gaussian_kernel_matrix += self.alpha * torch.eye(t1.size(1), device=t1.device, dtype=t1.dtype).unsqueeze(0)
         
        return gaussian_kernel_matrix
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(lengthscale={self.lengthscale.item()}, "
                f"variance={self.variance.item()})")      
        
        
#---------------------------------------------------------------------
        
      
class GaussianProcessPriorMaison(nn.Module):
    """Prior Processus Gaussien pour les variables latentes z_{1:T}.
    This class implements a Gaussian Process prior for the latent variables z_{1:T}.
    """
    
    def __init__(self,
        # z_dimension = 1,  # we assume z_dimension = 1 for now
        kernel = None,  # Kernel to use for the Gaussian Process
        mean_function = None,  # Mean function for the Gaussian Process
        ):
        
        super(GaussianProcessPriorMaison, self).__init__()
        
        # assert z_dimension == 1, "the code is not ready for z_dimension > 1 yet, sorry"
        
        if kernel is None:
            self.kernel = GaussianKernel()
        else:
            self.kernel = kernel
            
        if mean_function is None:
            self.mean_function = GPNullMean()
        else:
            self.mean_function = mean_function
            
    def forward(self, t):
        """Forward pass of the Gaussian Process prior.
        
        Args:
            t (torch.Tensor): Input tensor of shape (batch_size, sequence_length, 1)
        
        Returns:
            mean (torch.Tensor): Mean of the prior distribution of shape (batch_size, sequence_length, z_dimension=1)
                - computed using the mean function
            covariance (torch.Tensor): Covariance matrix of the prior distribution of shape (batch_size, sequence_length, sequence_length)
                - computed using the kernel
            torch.distributions.MultivariateNormal: Multivariate normal distribution representing the prior over z_{1:T}
                - mean is computed using the mean function, shape (batch_size, sequence_length, z_dimension=1)
                - covariance is computed using the kernel, shape (batch_size, sequence_length, sequence_length)
        NB : z_dimension = 1 in this first implementation.
        """
        
        assert t.dim() == 3, "In GPPrior, Input tensor must have shape (batch_size, sequence_length, 1)"
        assert t.size(-1) == 1, "In GPPrior, Input tensor must have last dimension = 1 (this is a set of times)"
        
        mean = self.mean_function(t) # (B, N, 1) => (B, N, Dz=1)
        covariance = self.kernel(t, t)  # (B, N, N)
        
        # instantiate the multivariate normal distribution
        prior_distribution = torch.distributions.MultivariateNormal(mean.squeeze(-1), covariance)
        
        return mean, covariance, prior_distribution
        
    def __repr__(self):
        msg = (f"{self.__class__.__name__}(kernel={self.kernel.__class__.__name__}, "
                f"mean_function={self.mean_function.__class__.__name__})")
        msg += f"\nKernel: {self.kernel}"
        msg += f"\nMean Function: {self.mean_function}"
        return msg



#---------------------------------------------------------------------
# LOSS FUNCTION
#---------------------------------------------------------------------

#
# Just to check....
#

def kl_maison(mu_0, sigma_0, mu_1, sigma_1):
    """Ugly KL divergence implementation between two multivariate normal distributions.
    
    Inputs:
        mu_0 (torch.Tensor): Mean of the first distribution (shape: (batch_size, sequence_length)
        sigma_0 (torch.Tensor): Covariance matrix of the first distribution (shape: (batch_size, sequence_length, sequence_length))
        mu_1 (torch.Tensor): Mean of the second distribution (shape: (batch_size, sequence_length))
        sigma_1 (torch.Tensor): Covariance matrix of the second distribution (shape: (batch_size, sequence_length, sequence_length))
        
    Returns:
        torch.Tensor: KL divergence between the two distributions (scalar)
        
    NB : this is a very ugly implementation, but it works for now. Next step is to be smart using Cholesky decomposition instead of linalg.inv
    """
    
    assert mu_0.shape == mu_1.shape, "mu_0 and mu_1 must have the same shape"
    assert sigma_0.shape == sigma_1.shape, "sigma_0 and sigma_1 must have the same shape"
    
    # compute the KL divergence - torch.einsum is my friend : https://ejenner.com/post/einsum/
    trace = torch.einsum('...ii->...', torch.linalg.inv(sigma_1) @ sigma_0)  # (batch_size)    
    outer_mu = torch.einsum('...i,...j->...ij', mu_1-mu_0, mu_1-mu_0)  # (batch_size, sequence_length, sequence_length)
    mahalanobis = torch.linalg.inv(sigma_1) @ outer_mu  # (batch_size, sequence_length, sequence_length)
    mahalanobis = torch.einsum('...ii->...', mahalanobis)  # (batch_size)
    logdet = torch.log(torch.det(sigma_1)) - torch.log(torch.det(sigma_0))  # (batch_size)
    kl_maison_value = 0.5 * (trace + mahalanobis - mu_0.shape[-1] + logdet)  # (batch_size)
    
    return kl_maison_value.mean()  # return the mean KL divergence over the batch

#----------------------------------------------------------------------
# Variational Lower Bound (VLB)
#----------------------------------------------------------------------

def vlb(q_phi, p_theta_x, p_theta_z, x_samples):
    """Variational Lower Bound (VLB) for the model.

    Args:
        q_phi (torch.distributions.MultivariateNormal): Encoder distribution q_phi(z_{1:T}|x_{1:T}).
        p_theta_x (torch.distributions.MultivariateNormal): Decoder distribution p_{\theta_x}(x_{1:T}|z_{1:T}).
        p_theta_z (torch.distributions.MultivariateNormal): Gaussian Process prior distribution p_{\theta_z}(z_{1:T}).
        z_sample (torch.Tensor): Sampled latent variables z_{1:T} from q_phi.
        x_samples (torch.Tensor): Sampled observations x_{1:T} from p_theta_x (K, B, L, Dx) : K samples
        
    Returns:
    kl_divergence (torch.Tensor): The KL divergence between q_phi and p_theta_z, computed by torch.distributions.kl.kl_divergence.
    kl_analytique (torch.Tensor): The KL divergence between q_phi and p_theta_z, computed using the custom kl_maison function (analytical KL divergence between Gaussian distributions).
    reconstruction_loss (torch.Tensor): Reconstruction loss, computed as the log likelihood of x_sample under p_theta_x.
    vlb_value: (torch.Tensor): The variational lower bound (VLB) value, computed as the difference between the reconstruction loss and the KL divergence.
    """
       
    # compute reconstruction loss
    log_probs = p_theta_x.log_prob(x_samples) # (K, B, L)
    reconstruction_loss = log_probs.sum(-1).mean()  # sum over the sequence length and mean over the batch and over the K samples
    
    # compute KL divergence
    kl_divergence = torch.distributions.kl.kl_divergence(q_phi, p_theta_z).mean()  # average over the batch
    
    # kl maison
    kl_analytique = kl_maison(
        mu_0=q_phi.mean, 
        sigma_0=q_phi.covariance_matrix, 
        mu_1=p_theta_z.mean, 
        sigma_1=p_theta_z.covariance_matrix
    )
    
    # compute the variational lower bound (VLB)
    vlb = reconstruction_loss - kl_divergence  # VLB = E_q[log p_theta_x(x|z)] - D_KL(q_phi(z|x) || p_theta_z(z))
    
    return kl_divergence, kl_analytique, reconstruction_loss, vlb















#---------------------------------------------------------------------
#----------------------------------------------------------------------
#
# TESTS
#
#----------------------------------------------------------------------
#---------------------------------------------------------------------


if __name__ == "__main__":
    # run some tests to check the implementation of the Encoder and Decoder
    seed_everything(42)
    
    # UTILITIES TESTS
    print("Testing Make MLP...")
    sequence_length = 10
    x_dimension = 5
    z_dimension = 1
    n_layers = 3
    inter_dim = 128
    activation = nn.ReLU
    
    mlp = make_mlp(
        input_dim=sequence_length * x_dimension,
        output_dim=sequence_length * z_dimension,  # output is a vector of length sequence_length * x_dimension
        n_layers=n_layers,
        inter_dim=inter_dim,
        activation=activation
    )
    print(mlp)
    
    # ENCODER TESTS  
    print(f"\nTest Encoder 0 : instantiation")
    sequence_length = 10
    x_dimension = 5
    z_dimension = 1
    n_layers = 3
    inter_dim = 128
    activation = nn.ReLU
    
    encoder = Encoder(
        sequence_length=sequence_length,
        x_dimension=x_dimension,
        z_dimension=z_dimension,
        n_layers=n_layers,
        inter_dim=inter_dim,
        activation=activation
    )
    
    print(encoder)
    
    print("\nTest Encoder 1 : forward pass with batch dimension")
    x = torch.randn(16, sequence_length, x_dimension)  # batch_size=2
    print(f"Input shape: {x.shape}")
    mu, sigma, q_phi = encoder(x)
    print(f"Output mu shape: {mu.shape}")
    print(f"Output sigma shape: {sigma.shape}")
    print(f"Output q_phi: {q_phi}")
    print(f"q_phi batch shape: {q_phi.batch_shape}")
    print(f"q_phi event shape: {q_phi.event_shape}")
    sample_z = q_phi.rsample()
    print(f"Sampled z shape: {sample_z.shape}")
    
    # DECODER TESTS
    print("\nTest Decoder 0 : instantiation...")
    
    decoder = GaussianDecoder(
        sequence_length=sequence_length,
        x_dimension=x_dimension,
        z_dimension=z_dimension,
        n_layers=n_layers,
        inter_dim=inter_dim,
        activation=activation
    )
    
    print(decoder)
    
    print(f"\nTest Decoder 1 : forward pass with batch dimension")
    z = torch.randn(16, sequence_length, z_dimension)  # batch_size=16
    print(f"Input shape: {z.shape}")
    mu_x, logvar_x, p_theta_x = decoder(z)
    print(f"Output mu_x shape: {mu_x.shape}")
    print(f"Output logvar_x shape: {logvar_x.shape}")
    print(f"Output p_theta_x: {p_theta_x}")
    print(f"p_theta_x batch shape: {q_phi.batch_shape}")
    print(f"p_theta_x event shape: {q_phi.event_shape}")
    
    sample_x = p_theta_x.rsample()
    print(f"Sampled x shape: {sample_x.shape}")
    
    print(f"\nTest Decoder 2 : testing log_probability of Decoder")
    print(f"log_probability of sampled x (shape): {p_theta_x.log_prob(sample_x).size()}")
    
    # GP PRIOR TESTS
    print("\nTest GPNullMean...")
    B, N, Z = 16, 250, 3 # batch_size, sequence_length, z_dimension
    gp_null_mean = GPNullMean(z_dimension=Z)
    t = torch.randn(B, N, 1)  # batch_size=16
    print(f"Input shape: {t.shape}")
    gp_mean_output = gp_null_mean(t)
    print(f"Output shape: {gp_mean_output.shape}")
    print(f"Output unique values: {gp_mean_output.unique()}")  # should be all zeros
    
    # CAUCHY KERNEL TESTS
    print("\nTest CauchyKernel...")
    cauchy_kernel = CauchyKernel()
    t1 = torch.randn(B, N, 1)  # batch_size=16
    t2 = torch.randn(B, N, 1)  # batch_size=16
    print(f"Input shapes: t1={t1.shape}, t2={t2.shape}")
    kernel_output = cauchy_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    
    try:
        torch.linalg.cholesky(kernel_output)
        print("Kernel matrix is positive definite.")
    except RuntimeError:
        print("Kernel matrix is NOT positive definite.")
    
    # TEST GP PRIOR
    print("\nTest GaussianProcessPriorMaison...")
    gp_prior = GaussianProcessPriorMaison()
    print(gp_prior)
    
    print(f"\nTest 1 avec 1ere dimension de t")
    N = 500  # sequence_length
    B = 1  # batch_size
    t = torch.randn(B, N, 1)  # batch_size=16
    print(f"Input shape (B, N, 1): {t.shape}")
    _, _, gp_prior_output = gp_prior(t)
    print(f"Output mean shape: {gp_prior_output.loc.shape}")
    print(f"Output covariance shape: {gp_prior_output.covariance_matrix.shape}")
    print(f"Batch shape: {gp_prior_output.batch_shape}")
    print(f"Event shape: {gp_prior_output.event_shape}")
    print(f"Sampled z shape: {gp_prior_output.rsample().shape}")
    
    print(f"\nTest 2 avec 2e dimension de t")
    N = 250  # sequence_length
    B = 32  # batch_size
    t = torch.randn(B, N, 1)  # batch_size=16
    print(f"Input shape: {t.shape}")
    _, _, gp_prior_output = gp_prior(t)
    print(f"Output mean shape: {gp_prior_output.loc.shape}")
    print(f"Output covariance shape: {gp_prior_output.covariance_matrix.shape}")
    print(f"Batch shape: {gp_prior_output.batch_shape}")
    print(f"Event shape: {gp_prior_output.event_shape}")
    print(f"Sampled z shape: {gp_prior_output.rsample().shape}") 
    
    # TEST LOSS FUNCTION
    print("\nTest VLB...")
    # we need to instantiate the Encoder and Decoder first
    B, L, Dx = 32, 5, 10  # batch_size, sequence_length, x_dimension
    sequence_length = L
    x_dimension = Dx
    z_dimension = 1  # we assume z_dimension = 1 for now
    
    encoder = Encoder(
        sequence_length=sequence_length,
        x_dimension=x_dimension,
        z_dimension=z_dimension,
        # n_layers=n_layers,
        # inter_dim=inter_dim,
        # activation=activation
    )   
    
    decoder = GaussianDecoder(
        sequence_length=sequence_length,
        x_dimension=x_dimension,
        z_dimension=z_dimension,
        # n_layers=n_layers,
        # inter_dim=inter_dim,
        # activation=activation
    )
    
    gp_prior = GaussianProcessPriorMaison(
        # z_dimension=z_dimension,
        kernel=CauchyKernel(),
        mean_function=GPNullMean(z_dimension=z_dimension),
    )
    
    x = torch.randn(B, L, Dx)
    print(f"Input shape x: {x.shape}")
    
    mu, sigma, q_phi = encoder(x)  # (B, L, Dz=1)
    print(f"Encoder distribution q_phi: {q_phi}")
    print(f"q_phi batch shape: {q_phi.batch_shape}")
    print(f"q_phi event shape: {q_phi.event_shape}")
    
    z_sample = q_phi.rsample().unsqueeze(2)  # (B, L, Dz=1)
    print(f"Sampled z shape: {z_sample.shape}")
    mu_x, logvar_x, p_theta_x = decoder(z_sample)  # (B, L, Dx)
    K = 3 # number of samples to draw from the decoder distribution
    x_samples = p_theta_x.rsample((K,))  # (B, L, Dx, K)
    print(f"Sampled x shape: {x_samples.shape}")
    
    print(f"Decoder distribution p_theta_x: {p_theta_x}")
    print(f"p_theta_x batch shape: {p_theta_x.batch_shape}")
    print(f"p_theta_x event shape: {p_theta_x.event_shape}")
    
    t = torch.randn(B, L, 1)  # batch_size=16
    print(f"Input shape for GP prior: {t.shape}")
    _, _, p_theta_z = gp_prior(t)  # (B, L, Dz=1)
    print(f"GP prior distribution p_theta_z: {p_theta_z}")
    print(f"p_theta_z batch shape: {p_theta_z.batch_shape}")
    print(f"p_theta_z event shape: {p_theta_z.event_shape}")
    
    kl, kl_maison_value, reco_loss, vlb = vlb(q_phi, p_theta_x, p_theta_z, x_samples)
    loss = -vlb
    reco_loss = -reco_loss
    print(f"KL divergence: {kl.item()}")
    print(f"KL divergence (maison): {kl_maison_value.item()}")
    print(f"Reconstruction loss: {reco_loss.item()}")
    print(f"VLB loss: {loss.item()}")