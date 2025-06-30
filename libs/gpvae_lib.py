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
            output_dim=sequence_length,
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
                
        self.diagonal_mlp = make_mlp(
            input_dim=sequence_length * x_dimension,
            output_dim=sequence_length,  # output is a vector of length sequence_length (* z_dim = 1)
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
                                       
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
            
        return L, L @ L.transpose(-1, -2)
    
    
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
        assert sequence_length == self.sequence_length, f"Incorrect sequence length passed to Encoder. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
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
    

class CauchyKernel(nn.Module):
    """Cauchy kernel for Gaussian Process.
    """
    
    def __init__(self):
        super(CauchyKernel, self).__init__()
        
        # the parameters of the kernel are learnable
        self.lengthscale = nn.Parameter(torch.tensor(1.0))  # learnable lengthscale parameter       
        self.variance = nn.Parameter(torch.tensor(1.0))  # learnable variance parameter
    
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
        return self.variance**2 * ( 1 + (diff / self.lengthscale)**2 )**(-1)  # (B, N, N) kernel matrix
    
      
class GaussianProcessPriorMaison(nn.Module):
    """Prior Processus Gaussien pour les variables latentes z_{1:T}.
    This class implements a Gaussian Process prior for the latent variables z_{1:T}.
    """
    
    def __init__(self,
        z_dimension = 1,  # we assume z_dimension = 1 for now
        kernel = None,  # Kernel to use for the Gaussian Process
        mean_function = None,  # Mean function for the Gaussian Process
        ):
        pass




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
    
    # GP PRIOR TESTS
    print("\nTest GPNullMean...")
    B, N, Z = 16, 50, 3 # batch_size, sequence_length, z_dimension
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