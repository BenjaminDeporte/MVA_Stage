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
from tqdm import tqdm


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
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 ):
        """_summary_

        Args:
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(EncoderMean, self).__init__()
        
        assert n_layers > 0, "n_layers must be greater than 0 !"
        
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
                
        self.mlp = make_mlp(
            input_dim=self.x_dimension,
            output_dim=self.z_dimension,  # output is a vector of length sequence_length * z_dimension
            n_layers=self.n_layers,
            inter_dim=self.inter_dim,
            activation=self.activation
        )
        
    
    def forward(self, input):
        """
        Forward pass of the EncoderMean.
        Args:
            x (torch.Tensor): Input tensor (..., N, x_dimension) (possibly N=1, ie a single point)
        Returns:
            z (torch.Tensor): Output tensor (..., z_dimension, N)
        """
        
        assert input.size(-1) == self.x_dimension, f"Wrong x_dim in EncoderMean. Input tensor must have shape (..., {self.x_dimension})"
        assert input.dim() >= 2, "Input tensor must have at least 2 dimensions (sequence_length -possibly 1- and x_dimension)"
        
        output = self.mlp(input)  # (..., N, Dz)
        output = torch.transpose(output,-1, -2)  # (..., Dz, N)
            
        return output  # (..., Dz, N) : batch of Dz independent but not identical Gaussians of dimension N. (can have N=1).
    
    def __repr__(self):
        return (f"{self.__class__.__name__}, "
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
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 alpha = 1e-3,  # small positive constant to ensure positive definiteness of the kernel matrix
                 ):
        """_summary_

        Args:
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(EncoderCovariance, self).__init__()
        
        assert n_layers > 0, "n_layers must be greater than 0 !"
        
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
        self.alpha = float(alpha)  # small positive constant to ensure positive definiteness of the kernel matrix
        
        # this network to get the diagonal elements of the covariance matrix
        self.diagonal_mlp = make_mlp(
            input_dim=self.x_dimension,
            output_dim=self.z_dimension,
            n_layers=self.n_layers,
            inter_dim=self.inter_dim,
            activation=activation
        )
        
        # this network to get Dz elements that will be the lower triangular part (without the diagonal) of the covariance matrix                
        self.full_matrix_mlp = make_mlp(
            input_dim=self.x_dimension,
            output_dim=self.z_dimension,
            n_layers=self.n_layers,
            inter_dim=self.inter_dim,
            activation=activation
        )
        
    
    def forward(self, x):
        """
        Forward pass of the EncoderCovariance.
        Args:
            x (torch.Tensor): Input tensor (..., N, x_dimension) 
                - N can be 1 (ie a single point) or the sequence length. Must be specified.

        Returns:
            L (torch.Tensor): Lower triangular matrix of shape (..., z_dimension, N, N) 
                L can be used as scale_tril in tf.distributions.MultivariateNormal
            C (torch.Tensor): Covariance matrix of shape (..., z_dimension, N, N) 
                C can be used as covariance_matrix in tf.distributions.MultivariateNormal. but this creates lots of numerical instabilities.
            C is computed as L @ L^T, where L is the lower triangular matrix. (Cholesky decomposition)
        """
        
        # basic check of shape of input x
        assert x.size(-1) == self.x_dimension, f"Wrong x_dim in EncoderCovariance. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions (sequence_length -possibly 1- and x_dimension)"
        
        # input is a tensor of shape (..., N, Dx)
        N = x.size(-2)  # sequence length
        
        # Compute the diagonal elements of the covariance matrix
        D = self.diagonal_mlp(x)  # out : (..., N, Dz)
        D = torch.transpose(D, -1, -2)  # out : (..., Dz, N)
        D = torch.exp(D) # out : (..., Dz, N). Ensure entries > 0.
        D = torch.diag_embed(D) # shape (..., Dz, N, N)
        
        # Get the elements outside the diagonal
        M = self.full_matrix_mlp(x) # shape (..., N, Dz)
        M = torch.transpose(M, -1, -2)  # shape (..., Dz, N)
        M1 = M.unsqueeze(-1)  # shape (..., Dz, N, 1)
        M2 = M.unsqueeze(-2)  # shape (..., Dz, 1, N)
        M = torch.matmul(M1, M2)  # shape (..., Dz, N, N)
        M = torch.tril(M, diagonal=-1)  # Keep only the lower triangular part of the matrix (excluding the diagonal) (..., Dz, N, N))
        
        # Assemble the lower triangular matrix L
        L = D + M # (..., Dz, N, N)
        
        # Assemble the covariance matrix C
        C = L @ L.transpose(-1, -2)  # C = L @ L^T # (..., Dz, N, N)
        C += self.alpha * torch.eye(N, device=x.device) # Add a small value to the diagonal for numerical stability (use broadcasting)
            
        return L, C # (..., Dz, N, N) Covariance matrix C
    
    def __repr__(self):
        return (f"{self.__class__.__name__}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__})"
                f"alpha={self.alpha:.3e})")


# brick 1 : Encoder q_phi

class Encoder(nn.Module):
    """Encoder module. 
    - Takes a sequence of length T of observations x_{1:T}, encodes them
    into a set of Dz independent but not identical Gaussians z_{1:T} of dimension T.
    - Computes the parameters of the multivariate normal distribution q_phi(z_{1:T}|x_{1:T}):
        - mu_phi(x_{1:T}): mean of the approximate posterior distribution
        - Sigma_phi(x_{1:T}): covariance matrix the approximate posterior distribution
    - Outputs the parameters and the distribution itself.
    """
    
    def __init__(self,                  
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,):
        
        super(Encoder, self).__init__()
        
        # self.sequence_length = sequence_length
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension
        self.n_layers = n_layers
        self.inter_dim = inter_dim
        
        self.encoder_mean = EncoderMean(
            x_dimension=self.x_dimension,
            z_dimension=self.z_dimension,
            n_layers=self.n_layers,
            inter_dim=self.inter_dim,
            activation=activation,
        )
        
        self.encoder_covariance = EncoderCovariance(
            x_dimension=self.x_dimension,
            z_dimension=self.z_dimension,
            n_layers=self.n_layers,
            inter_dim=self.inter_dim,
            activation=activation,
        )
    
    def forward(self, x):
        """
        Forward pass of the Encoder. The output is Dz independent but not identical Gaussians, of size the length of the sequence (either 1 or (..., N)).
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., N, x_dimension)
                - N can be 1 (ie a single point) or the sequence length. Must be specified.
            
        Returns:
            mu (torch.Tensor): Mean of the approximate posterior distribution of shape (..., Dz)
            Sigma (torch.Tensor): Covariance matrix of the approximate posterior distribution of shape (..., Dz, Dz)
            q_phi (torch.distributions.MultivariateNormal): Multivariate normal distribution with parameters mu:
                - if x has shape (Dx) (ie one point): MVN(mu, sigma) avec mu (Dz,1), sigma (Dz, 1, 1) => batch_shape = Dz, event_shape = 1.   
                    => q_phi is a MVN of Dz independent but not identical gaussians of dimension 1.
                - if x has shape (..., N, Dx) : MVN(mu, sigma) avec mu (...,Dz, N), sigma (..., Dz, N, N) => batch_shape = (..., Dz), event_shape = (N).
                    => q_phi is a MVN of(..., Dz) independent but not identical gaussians of dimension N.
        """
        
        # basic check of x shape
        assert x.size(-1) == self.x_dimension, f"Incorrect x_dimension passed to Encoder. Input tensor must have shape (..., {self.x_dimension})"
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions (sequence_length -possibly 1- and x_dimension)"
        
        # compute parameters of the approximate posterior distribution
        mu_phi = self.encoder_mean(x)  # (..., Dz, N)
        lower_phi, sigma_phi = self.encoder_covariance(x)  # (..., Dz, N, N)
        
        # instantiate the multivariate normal distribution
        # prefer instanciation with lower triangular matrix, less numerical instabilities
        q_phi = torch.distributions.MultivariateNormal(loc=mu_phi, scale_tril=lower_phi)
        
        return mu_phi, sigma_phi, q_phi        
    
    def __repr__(self):
        return (f"{self.__class__.__name__},"
            f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
            f"n_layers={self.encoder_mean.n_layers}, inter_dim={self.encoder_mean.inter_dim}, "
            f"activation={self.encoder_mean.activation.__name__})" 
            f"\nEncoderMean: {self.encoder_mean}"
            f"\nEncoderCovariance: {self.encoder_covariance}")


#--------------------------------------------------------------------------
# DECODER
#--------------------------------------------------------------------------

# brick 0.3 :Decoder Mean

class DecoderMean(nn.Module):
    """ Neural Net to compute the mean of the Gaussian Decoder distribution.
    """
    
    def __init__(self,
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 ):
        """_summary_

        Args:
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(DecoderMean, self).__init__()
        
        assert n_layers > 0, "n_layers must be greater than 0"
        
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
        
        self.mlp = make_mlp(
            input_dim=z_dimension,
            output_dim=x_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
    
    def forward(self, z):
        """
        Forward pass of the DecoderMean.
        Args:
            z (torch.Tensor): Input tensor of shape (..., z_dimension)
        Returns:
            torch.Tensor: mu_x : Output tensor of shape (..., x_dimension)
        """
              
        # mini check
        assert z.size(-1) == self.z_dimension, f"Incorrect latent dim. Input tensor z must have shape (..., {self.z_dimension})"
            
        return self.mlp(z)  # (..., Dx)
    
    def __repr__(self):
        return (f"{self.__class__.__name__} "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__})")
        
# brick 0.4 : Decoder Covariance

class DecoderCovariance(nn.Module):
    """ Neural Net to compute the covariance of the Gaussian Decoder distribution.
    """
    
    alpha = 1e-6  # small value to ensure numerical stability in the covariance matrix
    
    def __init__(self,
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU,
                 ):
        """_summary_

        Args:
            x_dimension (_type_, optional): Dimension des observations. Defaults to 1.
            z_dimension (_type_, optional): Dimension de l'espace latent. Defaults to 1.
            n_layers (int, optional): Nombre total de layers du MLP. Defaults to 3.
            inter_dim (int, optional): Nombre de neurones des hidden layers. Defaults to 128.
            activation (_type_, optional): Activation. Defaults to nn.ReLU. NB : la dernière couche n'a pas d'activation.
        """
        super(DecoderCovariance, self).__init__()
        
        assert n_layers > 0, "n_layers must be greater than 0"
        
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
        
        # this network to get the diagonal elements of the covariance matrix
        self.diagonal_mlp = make_mlp(
            input_dim=self.z_dimension,
            output_dim=self.x_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
        
        # this to get the lower triangular part (without the diagonal) of the covariance matrix
        self.full_matrix_mlp = nn.Sequential(
            make_mlp(
                input_dim=self.z_dimension,
                output_dim=self.x_dimension * self.x_dimension,
                n_layers=n_layers,
                inter_dim=inter_dim,
                activation=activation
                ),
            nn.Unflatten(-1, (self.x_dimension, self.x_dimension))  # Reshape the output to (Dx, Dx)
        )
    
    def forward(self, z):
        """
        Forward pass of the DecoderCovariance.
        Args:
            z (torch.Tensor): Input tensor of shape (..., z_dimension)
        Returns:
            L (torch.Tensor): Lower triangular matrix of shape (..., x_dimension, x_dimension) 
                L can be used as scale_tril in tf.distributions.MultivariateNormal
            C (torch.Tensor): Covariance matrix of shape (..., x_dimension, ...x_dimension) 
                C can be used as covariance_matrix in tf.distributions.MultivariateNormal. but this creates lots of numerical instabilities.
            C is computed as L @ L^T, where L is the lower triangular matrix. (Cholesky decomposition)
        """
        
        # mini check of z dimension
        assert z.size(-1) == self.z_dimension, f"Incorrect latent dim. Input tensor z must have shape (..., {self.z_dimension})"
                   
        # Compute the diagonal elements of the covariance matrix
        D = self.diagonal_mlp(z)  # out : (..., Dx)
        D = torch.exp(D) # out : (..., Dx). Ensure entries > 0.
        D = torch.diag_embed(D) # shape (..., Dx, Dx)
        
        # Get the elements outside the diagonal
        M = self.full_matrix_mlp(z) # shape (..., Dx, Dx)
        M = torch.tril(M, diagonal=-1)  # Keep only the lower triangular part of the matrix (excluding the diagonal) (..., Dx, Dx)
        
        # Assemble the lower triangular matrix L
        L = D + M # (..., Dx, Dx)
        
        # Assemble the covariance matrix C
        C = L @ L.transpose(-1, -2)  # C = L @ L^T # (..., Dz, Dz)
        C += self.alpha * torch.eye(self.x_dimension, device=z.device) # Add a small value to the diagonal for numerical stability (use broadcasting)
            
        return L, C # (..., Dx, Dx) Covariance matrix C
    
    def __repr__(self):
        return (f"{self.__class__.__name__}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__})")
 
# brick 2 : decoder p_\{theta_x}

class GaussianDecoder(nn.Module):
    """Decoder module. Takes a sequence of latent variables and computes the parameters of the
    obervation model p_{\theta_x}. Here, the observation model is a Gaussian distribution,
    and an observation x_t depends only on the latent variable z_t at time t. Therefore,
    the computed covariance matrix is diagonal so the observations are independent.
    """
    
    alpha = 1e-12  # small value to ensure numerical stability in the covariance matrix

    def __init__(self,
                 x_dimension = 1,
                 z_dimension = 1, 
                 n_layers = 3,
                 inter_dim = 128,
                 activation = nn.ReLU
                 ):
        
        super(GaussianDecoder, self).__init__()
        
        self.x_dimension = x_dimension
        self.z_dimension = z_dimension
        
        self.decoder_mean = DecoderMean(
            x_dimension=x_dimension,
            z_dimension=z_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation,
        )
        
        self.decoder_covariance = DecoderCovariance(
            x_dimension=x_dimension,
            z_dimension=z_dimension,
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation,
        )
    
    def forward(self, z):
        """Takes a set of variables z (..., Dz)
        
        Computes the parameters of the observation model p_{\theta_x}(x|z):
            - mu_x(z): mean of the observation model (..., Dx)
            - sigma_x(z): covariance matrix of the observation model (..., Dx, Dx)
            - p_theta_x: the observation model itself. (instanciated with the lower triangular matrix L)
        """
        
        # manage shape of z
        assert z.size(-1) == self.z_dimension, f"Incorrect x_dimension passed to Decoder. Input tensor must have shape (..., {self.z_dimension})"
        
        # compute parameters of the observation model        
        mu_x = self.decoder_mean(z)  # (..., Dx)
        lower_x, sigma_x = self.decoder_covariance(z)  # (..., Dx, Dx)
        
        # instantiate the multivariate normal distribution
        # prefer instanciation with lower triangular matrix, less numerical instabilities
        p_theta_x = torch.distributions.MultivariateNormal(loc=mu_x, scale_tril=lower_x)
        
        return mu_x, sigma_x, p_theta_x
    
    def __repr__(self):
        msg = (f"{self.__class__.__name__}, "
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
    """ Neural Net to compute the mean of one univariate Gaussian Process
    This is a null mean function, i.e. it returns a tensor of zeros...
    """
    
    def __init__(self):
        super(GPNullMean, self).__init__()
    
    def forward(self, t):
        """Forward pass of the GPNullMean.
        Just returns a tensor of zeros of shape identical to the input.
        
        Inputs:
            t (torch.Tensor): Input tensor of shape (..., N) - N is typically the sequence length.
        Returns:
            torch.Tensor: Output tensor of ZEROS of shape (..., N)
        """
        
        return torch.zeros_like(t, dtype=t.dtype, device=t.device)  # (..., N)
    
    def __repr__(self):
        return f"{self.__class__.__name__}"
    

# ----- Cauchy Kernel ---------------------------------------------
#
# HIGHLY UNSTABLE KERNEL !  Produces not definite positive matrices from time to time !
#

class CauchyKernel(nn.Module):
    """Cauchy kernel for Gaussian Process.
    NB : requires a small positive constant alpha to ensure positive definiteness of the kernel matrix.
    The lengthscale and variance parameters are learnable (nn.Parameter).
    """
    
    def __init__(self, lengthscale=1.0, variance=1.0, alpha=1e-3):
        super(CauchyKernel, self).__init__()
        
        # the parameters of the kernel are learnable
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))  # learnable lengthscale parameter       
        self.variance = nn.Parameter(torch.tensor(variance))  # learnable variance parameter (ie sigma**2)
        self.alpha = torch.tensor(alpha)  # small positive constant to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Gaussian kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (..., N) - N is typically the sequence length.
            t2 (torch.Tensor): Second set of time points (..., M)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (..., N, M)
        """
        
        assert t1.dim() == t2.dim(), "CauchyKernel object : Input tensors must have the same number of dimensions"
        
        if t1.dim() == 1:
            t1_b = t1.unsqueeze(-1)  # (N, 1)
            t2_b = t2.unsqueeze(0)   # (1, M)
            mat = 1 + ((t1_b - t2_b)**2 / self.lengthscale**2)  # (N, M)
        else:
            t1_b = t1.unsqueeze(-1) # (..., N, 1)
            t2_b = t2.unsqueeze(-2) # (..., 1, M) 
            mat = 1 + ((t1_b - t2_b)**2 / self.lengthscale**2)  # (..., N, M)
            
        # compute the Cauchy kernel matrix
        cauchy_kernel_matrix = self.variance / mat  # (..., N, M)
        
        if torch.equal(t1, t2):
            # If t1 and t2 are the same, the kernel matrix should be symmetric and positive definite
            # so we compute and return the Cholesky decomposition of the kernel matrix
            # to be used in forming the MultivariateNormal distribution, adding
            # a small value to the diagonal for numerical stability
            cauchy_kernel_matrix += self.alpha * torch.eye(t1.size(-1), device=t1.device, dtype=t1.dtype)
            try:
                L = torch.linalg.cholesky(cauchy_kernel_matrix)  # Cholesky decomposition to ensure positive definiteness
                return cauchy_kernel_matrix, L  # Return the kernel matrix and its Cholesky factor L
            except RuntimeError:
                # If the Cholesky decomposition fails, it means the matrix is not positive definite
                # We can return None or raise an error, depending on how we want to handle this case
                # print("Warning: Cholesky decomposition failed.")
                # print(f"Kernel : {gaussian_kernel_matrix}")
                # return gaussian_kernel_matrix, None  # Return the kernel matrix and None for L
                raise NameError("Cholesky decomposition of a supposedly PSD kernel matrix failed in CauchyKernel. Tolerance alpha is likely too low.") 
        else:
            # If t1 and t2 are different, do not try to compute the Cholesky decomposition
            return cauchy_kernel_matrix, None
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(lengthscale={self.lengthscale.item()}, "
                f"variance={self.variance.item()}), "
                f"alpha={self.alpha.item():.3e})")     
        

#----- Gaussian Kernel --------------------------------------------
#

class GaussianKernel(nn.Module):  
    """Gaussian kernel for one univariate Gaussian Process.
    The lengthscale and variance parameters are learnable (nn.Parameter).
    """
    
    # we need to add a small positive constant to ensure positive definiteness of the kernel matrix
    # 1e-3 is ok for time series up to 10,000 time points.
    # 1e-4 is ok for time series up to 2,880 time points. (ie 2 days @ 1 minute resolution).
    # 1e-4 is ok for time series up to 1,000 time points.
    # 1e-6 is ok for time series up to 100 time points.
    # the value can be decreased for shorter time series.
    # but it should not be too small, otherwise the Cholesky decomposition will fail.
    
    def __init__(self, lengthscale=1.0, variance=1.0, alpha=1e-3):
        super(GaussianKernel, self).__init__()
        
        # learnable parameters for the Gaussian kernel
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))  # learnable lengthscale parameter       
        self.variance = nn.Parameter(torch.tensor(variance))  # learnable variance parameter
        self.alpha = torch.tensor(alpha)  # tolerance to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Gaussian kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (..., N) - N is typically the sequence length.
            t2 (torch.Tensor): Second set of time points (..., M)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (..., N, M)
        """
        
        assert t1.dim() == t2.dim(), "GaussianKernel object : Input tensors must have the same number of dimensions"
        
        # compute the Gaussian kernel matrix
        if t1.dim() == 1:
            t1_b = t1.unsqueeze(-1)  # (N, 1)
            t2_b = t2.unsqueeze(0)   # (1, M)
            kernel = torch.exp(-0.5 * ((t1_b - t2_b) / self.lengthscale)**2)  # (N, M)
        else:
            t1_b = t1.unsqueeze(-1) # (...,N, 1)
            t2_b = t2.unsqueeze(-2) # (...,1, M)
            kernel = torch.exp(-0.5 * ((t1_b - t2_b) / self.lengthscale)**2)  # (..., N, M)
            
        gaussian_kernel_matrix = self.variance * kernel  # (..., N, M)

        if torch.equal(t1, t2):
            # If t1 and t2 are the same, the kernel matrix should be symmetric and positive definite
            # so we compute and return the Cholesky decomposition of the kernel matrix
            # to be used in forming the MultivariateNormal distribution, adding
            # a small value to the diagonal for numerical stability
            gaussian_kernel_matrix += self.alpha * torch.eye(t1.size(-1), device=t1.device, dtype=t1.dtype)
            try:
                L = torch.linalg.cholesky(gaussian_kernel_matrix)  # Cholesky decomposition to ensure positive definiteness
                return gaussian_kernel_matrix, L  # Return the kernel matrix and its Cholesky factor L
            except RuntimeError:
                # If the Cholesky decomposition fails, it means the matrix is not positive definite
                # We can return None or raise an error, depending on how we want to handle this case
                # print("Warning: Cholesky decomposition failed.")
                # print(f"Kernel : {gaussian_kernel_matrix}")
                # return gaussian_kernel_matrix, None  # Return the kernel matrix and None for L
                raise NameError("Cholesky decomposition of a supposedly PSD kernel matrix failed. Tolerance alpha is likely too low.") 
        else:
            # If t1 and t2 are different, do not try to compute the Cholesky decomposition
            return gaussian_kernel_matrix, None
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(lengthscale={self.lengthscale.item()}, "
                f"variance={self.variance.item()}), "
                f"alpha={self.alpha.item():.3e})")      
        
        
#---------------------------------------------------------------------
        
      
class GaussianProcessPriorMaison(nn.Module):
    """Prior Processus Gaussien pour les variables latentes z_{1:T}.
    This class implements a Gaussian Process prior for the latent variables z_{1:T}.
    """
    
    def __init__(self,
        z_dimension = 1,  # dimension of the latent variable z, ie number of different Gaussian Processes
        kernels_list = None,  # list of Kernels to use for the Gaussian Processes
        mean_functions_list = None,  # list of Mean functions to use for the Gaussian Processes
        alpha=1e-3,  # small positive constant to ensure positive definiteness of the kernel matrix
        ):
        
        super(GaussianProcessPriorMaison, self).__init__()
        
        self.alpha = torch.tensor(alpha)  # small positive constant to ensure positive definiteness of the kernel matrix
        
        # assert z_dimension == 1, "Currently, only one Gaussian Process is implemented. z_dimension must be 1."
        self.z_dimension = int(z_dimension)  # number of different Gaussian Processes²
        
        if kernels_list is None:
            self.kernels_list = [GaussianKernel(alpha=alpha)] * self.z_dimension  # default kernel is a Gaussian kernel with alpha
        else:
            self.kernels_list = kernel_list
            
        if mean_functions_list is None:
            self.mean_functions_list = [ GPNullMean() ] * self.z_dimension  # default mean function is a null mean function
        else:
            self.mean_functions_list = mean_functions_list
            
    def forward(self, t):
        """Forward pass of the Gaussian Process prior.
        
        Args:
            t (torch.Tensor): Input tensor of shape (..., N) - N is typically the sequence length.
        
        Returns:
            mean (torch.Tensor): Mean of the prior distribution of shape (..., Dz, N)
                - computed using the mean function
            covariance (torch.Tensor): Covariance matrix of the prior distribution of shape (..., Dz, N, N)
                - computed using the kernel
            torch.distributions.MultivariateNormal: Multivariate normal distribution representing the prior over z
                - mean is computed using the mean functions, shape (..., Dz, N)
                - covariance is computed using the kernel, shape (..., Dz, N, N)
                NB : the torch.distributions.MultivariateNormal is instantiated with the mean and the Cholesky
                decomposition of the covariance matrix to ensure numerical stability.
                => the MVN has batch_shape (..., Dz) and event_shape (N).
        """
        
        # compute the mean and covariance of the prior distribution
        times = t.unsqueeze(-1)  # (..., N, 1)
        times = torch.repeat_interleave(times, repeats=self.z_dimension, dim=-1) # (..., N, Dz)
        times = torch.transpose(times, -1, -2)  # (..., Dz, N)
        
        # means of the z_dim different Gaussian Processes
        # inits
        means = torch.zeros_like(times, dtype=t.dtype, device=t.device)  # (..., Dz, N)
        covariances = torch.zeros_like(times, dtype=t.dtype, device=t.device)  # (..., Dz, N)
        covariances = covariances.unsqueeze(-1)  # (..., Dz, N, 1)
        covariances = torch.repeat_interleave(covariances, repeats=times.size(-1), dim=-1)  # (..., Dz, N, N)
        Ls = torch.zeros_like(covariances, dtype=t.dtype, device=t.device)  # (..., Dz, N, N)
        
        # loop over the different Gaussian Processes
        for i in range(self.z_dimension):
            means[..., i, :] = self.mean_functions_list[i](times[..., i, :])  # (..., Dz, N)
            covariances[..., i, :, :], Ls[..., i, :, :] = self.kernels_list[i](times[..., i, :], times[..., i, :])  # (..., Dz, N, N)
        
        # if L is None:
        #     raise NameError("Cholesky decomposition of a supposedly PSD kernel matrix failed. Tolerance alpha is likely too low.")
        
        # instantiate the multivariate normal distribution
        # batch_shape = (..., Dz) and event_shape = (N)
        prior_distribution = torch.distributions.MultivariateNormal(loc=means, scale_tril=Ls) 
        
        return means, covariances, prior_distribution
        
    def __repr__(self):
        msg = (f"{self.__class__.__name__}"
               f"(z_dimension={self.z_dimension}, ")
            #    f"(kernel={self.kernel.__class__.__name__}, "
            #     f"mean_function={self.mean.__class__.__name__})")
        msg += f"\nKernels list: {self.kernels_list}"
        msg += f"\nMean Functions list: {self.mean_functions_list}"
        msg += f"\nAlpha: {self.alpha.item():.3e}"
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
    # kl_analytique = 0.0
    
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

#----------------------------------------------------------

def utilities_test():
    # UTILITIES TESTS
    print(f"*" * 10 + " UTILITIES TESTS " + "*" * 10)
    print("Testing Make MLP...")
    n_layers = 3
    inter_dim = 128
    activation = nn.ReLU
    
    mlp = make_mlp(
        input_dim=x_dimension,
        output_dim=z_dimension,  # output is a vector of length sequence_length * x_dimension
        n_layers=n_layers,
        inter_dim=inter_dim,
        activation=activation
    )
    print(mlp)
    
#-------------------------------------------------------

def encoder_tests():
    print(f"*" * 10 + " ENCODER TESTS " + "*" * 10)
    print(f"\nTest Encoder 0 : instantiation")
    n_layers = 3
    inter_dim = 128
    activation = nn.ReLU
    
    encoder = Encoder(
        x_dimension=x_dimension,
        z_dimension=z_dimension,
        n_layers=n_layers,
        inter_dim=inter_dim,
        activation=activation
    )
    
    print(encoder)
    
    print("\nTest Encoder 1 : forward pass with (Dx) dimension")
    x = torch.randn(x_dimension)
    print(f"Input shape: {x.shape}")
    x = x.unsqueeze(0)  # expand to (1, Dx) for batch processing
    print(f"Expanding x to (1, Dx) for batch processing... : {x.shape}")
    mu, sigma, q_phi = encoder(x)
    print(f"Output mu shape: {mu.shape}")
    print(f"Output sigma shape: {sigma.shape}")
    print(f"Output q_phi: {q_phi}")
    print(f"q_phi batch shape: {q_phi.batch_shape}")
    print(f"q_phi event shape: {q_phi.event_shape}")
    sample_z = q_phi.rsample()
    print(f"Sampled z shape: {sample_z.shape}")
    
    print("\nTest Encoder 2 : forward pass with (N, Dx) dimension")
    x = torch.randn(sequence_length, x_dimension)
    print(f"Input shape: {x.shape}")
    mu, sigma, q_phi = encoder(x)
    print(f"Output mu shape: {mu.shape}")
    print(f"Output sigma shape: {sigma.shape}")
    print(f"Output q_phi: {q_phi}")
    print(f"q_phi batch shape: {q_phi.batch_shape}")
    print(f"q_phi event shape: {q_phi.event_shape}")
    sample_z = q_phi.rsample()
    print(f"Sampled z shape: {sample_z.shape}")
    
    print("\nTest Encoder 3 : forward pass with (B, N, Dx) dimension")
    x = torch.randn(batch_size, sequence_length, x_dimension)  # batch_size=2
    print(f"Input shape: {x.shape}")
    mu, sigma, q_phi = encoder(x)
    print(f"Output mu shape: {mu.shape}")
    print(f"Output sigma shape: {sigma.shape}")
    print(f"Output q_phi: {q_phi}")
    print(f"q_phi batch shape: {q_phi.batch_shape}")
    print(f"q_phi event shape: {q_phi.event_shape}")
    sample_z = q_phi.rsample()
    print(f"Sampled z shape: {sample_z.shape}")
    
#-------------------------------------------------------

def decoder_tests():
    print(f"*" * 10 + " DECODER TESTS " + "*" * 10)
    print("\nTest Decoder 0 : instantiation...")
    n_layers = 3
    inter_dim = 128
    activation = nn.ReLU
    
    decoder = GaussianDecoder(
        x_dimension=x_dimension,
        z_dimension=z_dimension,
        n_layers=n_layers,
        inter_dim=inter_dim,
        activation=activation
    )
    
    print(decoder)
    
    print("\nTest Decoder 1 : forward pass with (Dz) dimension")
    z = torch.randn(z_dimension)
    print(f"Input shape: {z.shape}")
    mu_x, sigma_x, p_theta_x = decoder(z)
    print(f"Output mu_x shape: {mu_x.shape}")
    print(f"Output sigma_x shape: {sigma_x.shape}")
    print(f"Output p_theta_x: {p_theta_x}")
    print(f"\tp_theta_x batch shape: {p_theta_x.batch_shape}")
    print(f"\tp_theta_x event shape: {p_theta_x.event_shape}")
    sample_x = p_theta_x.rsample()
    print(f"Sampled x shape: {sample_x.shape}")
    
    print("\nTest Decoder 2 : forward pass with (N, Dz) dimension")
    z = torch.randn(sequence_length, z_dimension)
    print(f"Input shape: {z.shape}")
    mu_x, sigma_x, p_theta_x = decoder(z)
    print(f"Output mu_x shape: {mu_x.shape}")
    print(f"Output sigma_x shape: {sigma_x.shape}")
    print(f"Output p_theta_x: {p_theta_x}")
    print(f"\tp_theta_x batch shape: {p_theta_x.batch_shape}")
    print(f"\tp_theta_x event shape: {p_theta_x.event_shape}")
    sample_x = p_theta_x.rsample()
    print(f"Sampled x shape: {sample_x.shape}")
    
    print("\nTest Decoder 3 : forward pass with (B,N,Dx) dimension")
    z = torch.randn(batch_size, sequence_length, z_dimension)
    print(f"Input shape: {z.shape}")
    mu_x, sigma_x, p_theta_x = decoder(z)
    print(f"Output mu_x shape: {mu_x.shape}")
    print(f"Output sigma_x shape: {sigma_x.shape}")
    print(f"Output p_theta_x: {p_theta_x}")
    print(f"\tp_theta_x batch shape: {p_theta_x.batch_shape}")
    print(f"\tp_theta_x event shape: {p_theta_x.event_shape}")
    sample_x = p_theta_x.rsample()
    print(f"Sampled x shape: {sample_x.shape}")
    
    print(f"\nTest Decoder 4 : testing log_probability of Decoder")
    print(f"log_probability of sampled x (shape): {p_theta_x.log_prob(sample_x).size()}")
    
#-------------------------------------------------------

def gp_prior_tests():
    print(f"*" * 10 + " GP PRIOR TESTS " + "*" * 10)
    print("\nTest GPNullMean...")
    B, N, Z = 1, 2, 3 # batch_size, sequence_length, z_dimension
    gp_null_mean = GPNullMean()
    print(gp_null_mean)
    print()
    
    t = torch.randn(N)
    print(f"Input shape: {t.shape}")
    gp_mean_output = gp_null_mean(t)
    print(f"Output shape: {gp_mean_output.shape}")
    print(f"Output unique values: {gp_mean_output.unique()}")  # should be all zeros
    print()
    
    t = torch.randn(B, N)
    print(f"Input shape: {t.shape}")
    gp_mean_output = gp_null_mean(t)
    print(f"Output shape: {gp_mean_output.shape}")
    print(f"Output unique values: {gp_mean_output.unique()}")  # should be all zeros
    print()
    
    # TEST GP PRIOR
    print("\nTest GaussianProcessPriorMaison...")
    print(f"Dz = 3")
    Dz = 3  # number of different Gaussian Processes
    gp_prior = GaussianProcessPriorMaison(z_dimension=Dz)
    print(gp_prior)
    
    print(f"\nTest 1 avec 1ere dimension de t")
    N = 500  # sequence_length
    B = 1  # batch_size
    t = torch.randn(N)
    print(f"Input shape: {t.shape}")
    _, _, gp_prior_output = gp_prior(t)
    print(f"Output mean shape: {gp_prior_output.loc.shape}")
    print(f"Output covariance shape: {gp_prior_output.covariance_matrix.shape}")
    print(f"Batch shape: {gp_prior_output.batch_shape}")
    print(f"Event shape: {gp_prior_output.event_shape}")
    print(f"Sampled z shape: {gp_prior_output.rsample().shape}")
    
    print(f"\nTest 2 avec 2e dimension de t")
    N = 250  # sequence_length
    B = 32  # batch_size
    t = torch.randn(B, N)
    print(f"Input shape: {t.shape}")
    _, _, gp_prior_output = gp_prior(t)
    print(f"Output mean shape: {gp_prior_output.loc.shape}")
    print(f"Output covariance shape: {gp_prior_output.covariance_matrix.shape}")
    print(f"Batch shape: {gp_prior_output.batch_shape}")
    print(f"Event shape: {gp_prior_output.event_shape}")
    print(f"Sampled z shape: {gp_prior_output.rsample().shape}") 
    
#-------------------------------------------------------

def gaussian_kernel_tests():
    print(f"*" * 10 + " GAUSSIAN KERNEL TESTS " + "*" * 10)
    print("\nTest Gaussian Kernel...")
    
    gaussian_kernel = GaussianKernel()
    print(gaussian_kernel)
    print()
    
    # Test with same 1D input
    N = 4
    t1 = torch.randn(N)  # sequence_length
    t2 = t1  # same input
    print(f"Input shapes: t1 = t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = gaussian_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # test with different 1D inputs, same sequence lengths
    N = 16  # sequence_length
    t1 = torch.randn(N)  # sequence_length
    t2 = torch.randn(N)  # sequence_length
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = gaussian_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # Test with 1D input, and different sequence lengths
    N, M = 4, 8
    t1 = torch.randn(N)  # sequence_length
    t2 = torch.randn(M)  # sequence_length
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = gaussian_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")  # L is None if t1 != t2
    
    # test with same 2D inputs, same sequence lengths
    B = 16  # batch_size
    N, M = 3, 3  # sequence_length
    t1 = torch.randn(B, N)  # batch_size, sequence_length
    t2 = t1  # same input
    print(f"Input shapes: t1 = t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = gaussian_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # Test with different 2D inputs, same shape
    t1 = torch.randn(B, N)  # batch_size, sequence_length
    t2 = torch.randn(B, N)  # same input
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = gaussian_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # Test with 2D input, different sequence lengths
    t1 = torch.randn(B, N)
    t2 = torch.randn(B, M)
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = gaussian_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")  # L is None if t1 != t2
    
    # Brute force for positive definiteness
    print("\nBrute force check for positive definiteness...")
    gaussian_kernel = GaussianKernel(alpha=1e-3)
    print(gaussian_kernel)
    TESTS = int(1e+2)
    B = 32 # batch_size
    N = 1000  # sequence_length 2880 = 2 days @ 1 minute
    print(f"Running {TESTS} tests with B={B}, N={N}")
    failures = 0
    for i in tqdm(range(TESTS)):
        t1 = torch.randn(B,N)
        kernel_output, L = gaussian_kernel(t1, t1)  # same input
        if L is None:
            failures += 1
    print(f"All tests completed. - {TESTS - failures} passed, {failures} failed.")  
    
#---------------------------------------------------------------------
    
def cauchy_kernel_tests():
    print(f"*" * 10 + " CAUCHY KERNEL TESTS " + "*" * 10)
    print("\nTest Cauchy Kernel...")
    
    cauchy_kernel = CauchyKernel()
    print(cauchy_kernel)
    print()
    
    # Test with same 1D input
    N = 4
    t1 = torch.randn(N)  # sequence_length
    t2 = t1  # same input
    print(f"Input shapes: t1 = t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = cauchy_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # test with different 1D inputs, same sequence lengths
    N = 16  # sequence_length
    t1 = torch.randn(N)  # sequence_length
    t2 = torch.randn(N)  # sequence_length
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = cauchy_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # Test with 1D input, and different sequence lengths
    N, M = 4, 8
    t1 = torch.randn(N)  # sequence_length
    t2 = torch.randn(M)  # sequence_length
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = cauchy_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")  # L is None if t1 != t2
    
    # test with same 2D inputs, same sequence lengths
    B = 16  # batch_size
    N, M = 3, 3  # sequence_length
    t1 = torch.randn(B, N)  # batch_size, sequence_length
    t2 = t1  # same input
    print(f"Input shapes: t1 = t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = cauchy_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # Test with different 2D inputs, same shape
    t1 = torch.randn(B, N)  # batch_size, sequence_length
    t2 = torch.randn(B, N)  # same input
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = cauchy_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")
    
    # Test with 2D input, different sequence lengths
    t1 = torch.randn(B, N)
    t2 = torch.randn(B, M)
    print(f"Input shapes: t1 != t2, t1={t1.shape}, t2={t2.shape}")
    kernel_output, L = cauchy_kernel(t1, t2)
    print(f"Output shape: {kernel_output.shape}")
    print(f"L : {L.shape if L is not None else 'None'}")  # L is None if t1 != t2
    
    # Brute force for positive definiteness
    print("\nBrute force check for positive definiteness...")
    cauchy_kernel = CauchyKernel(alpha=1e-3)
    print(cauchy_kernel)
    TESTS = int(1e+2)
    B = 32 # batch_size
    N = 1000  # sequence_length 2880 = 2 days @ 1 minute
    print(f"Running {TESTS} tests with B={B}, N={N}")
    failures = 0
    for i in tqdm(range(TESTS)):
        t1 = torch.randn(B,N)
        kernel_output, L = cauchy_kernel(t1, t1)  # same input
        if L is None:
            failures += 1
    print(f"All tests completed. - {TESTS - failures} passed, {failures} failed.")  
  
#----------------------------------------------------------------    

if __name__ == "__main__":
    # run some tests to check the implementation of the Encoder and Decoder
    seed_everything(42)
    
    batch_size = 32
    sequence_length = 10
    x_dimension = 16
    z_dimension = 4
    print(f"Dx= {x_dimension}, Dz={z_dimension}, sequence_length={sequence_length}, batch_size={batch_size}")
    
    # # UTILITIES TESTS
    # utilities_test()
    
    # # ENCODER TESTS
    # encoder_tests()
    
    # # DECODER TESTS
    # decoder_tests()
    
    # # GAUSSIAN KERNEL TESTS
    gaussian_kernel_tests()
    
    # CAUCHY KERNEL TESTS
    cauchy_kernel_tests()
    
    # GP PRIOR TESTS
    # gp_prior_tests()
    

    

    

    
    # # TEST LOSS FUNCTION
    # print("\nTest VLB...")
    # # we need to instantiate the Encoder and Decoder first
    # B, L, Dx = 32, 5, 10  # batch_size, sequence_length, x_dimension
    # sequence_length = L
    # x_dimension = Dx
    # z_dimension = 1  # we assume z_dimension = 1 for now
    # print(f"Dx= {x_dimension}, Dz={z_dimension}, sequence_length={sequence_length}, batch_size={B}")
    
    # encoder = Encoder(
    #     x_dimension=x_dimension,
    #     z_dimension=z_dimension,
    #     # n_layers=n_layers,
    #     # inter_dim=inter_dim,
    #     # activation=activation
    # )   
    
    # decoder = GaussianDecoder(
    #     x_dimension=x_dimension,
    #     z_dimension=z_dimension,
    #     # n_layers=n_layers,
    #     # inter_dim=inter_dim,
    #     # activation=activation
    # )
    
    # gp_prior = GaussianProcessPriorMaison(
    #     # z_dimension=z_dimension,
    #     kernel=GaussianKernel(),
    #     mean=GPNullMean(),
    # )
    
    # x = torch.randn(B, L, Dx)
    # print(f"Input shape x: {x.shape}")
    
    # mu, sigma, q_phi = encoder(x)  # (B, L, Dz=1)
    # print(f"Encoder distribution q_phi: {q_phi}")
    # print(f"\tq_phi batch shape: {q_phi.batch_shape}")
    # print(f"\tq_phi event shape: {q_phi.event_shape}")
    
    # z_sample = q_phi.rsample()  # (B, L, Dz=1)
    # print(f"Sampled z shape: {z_sample.shape}")
    # mu_x, logvar_x, p_theta_x = decoder(z_sample)  # (B, L, Dx)
    # K = 3 # number of samples to draw from the decoder distribution
    # x_samples = p_theta_x.rsample((K,))  # (B, L, Dx, K)
    # print(f"Sampling {K} x's, shape: {x_samples.shape}")
    
    # print(f"Decoder distribution p_theta_x: {p_theta_x}")
    # print(f"\tp_theta_x batch shape: {p_theta_x.batch_shape}")
    # print(f"\tp_theta_x event shape: {p_theta_x.event_shape}")
    
    # t = torch.randn(B, L, 1)  # batch_size=16
    # print(f"Input shape for GP prior: {t.shape}")
    # _, _, p_theta_z = gp_prior(t)  # (B, L, Dz=1)
    # print(f"GP prior distribution p_theta_z: {p_theta_z}")
    # print(f"\tp_theta_z batch shape: {p_theta_z.batch_shape}")
    # print(f"\tp_theta_z event shape: {p_theta_z.event_shape}")
    
    # kl, kl_maison_value, reco_loss, vlb = vlb(q_phi, p_theta_x, p_theta_z, x_samples)
    # loss = -vlb
    # reco_loss = -reco_loss
    # print(f"KL divergence: {kl.item()}")
    # print(f"KL divergence (maison): {kl_maison_value.item()}")
    # print(f"Reconstruction loss: {reco_loss.item()}")
    # print(f"VLB loss: {loss.item()}")