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
    
    
#------------------------------------------------------------------------
#
# CNN Encoder
#
#------------------------------------------------------------------------
#
# --- Takes (B,T) images of shape (W=64, H=64, C=3)
# --- pre-encodes them into (B,T,Dz) vectors (x2 : one for mean, one for logvar)
#

class EncoderCNN(nn.Module):
    """
    Encoder CNN module.
    Takes images of shape (B, T, W=64, H=64, C=3)
    and encodes them into a latent space of dimension Dz.
    """
    def __init__(self, Dz=2, hidden_dim=64):
        super(EncoderCNN, self).__init__()
        self.Dz = Dz  # Dimension of the latent space
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
        # intermediate fully connected layer
        self.fc1 = nn.Linear(128 * 8 * 8, hidden_dim)
        # output mean and logvar of the latent space
        self.fc_mu = nn.Linear(hidden_dim, self.Dz)
        self.fc_logcovar = nn.Linear(hidden_dim, self.Dz)
    
    def forward(self, x):
        """
        Input: (B, T, W=64, H=64, C=3)
        Output: 
        - mu_z = (B, T, Dz) ... data for the mean of the gaussian posterior of z
        - log_var_z = (B, T, Dz) ... data for the covariance matrix of the gaussian posterior of z
        
        """
        # check input shape
        assert x.dim() == 5, f"Input shape must be (T, B, W, H, C), got {x.shape}"
        assert x.shape[2] == 64 and x.shape[3] == 64 and x.shape[4] == 3, \
            f"Input shape must be (T, B, 64, 64, 3), got {x.shape}"
        # manage shape
        B = x.shape[0]  # Batch dimension
        T = x.shape[1]  # Time dimension
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
        x = self.fc1(x)  # (T*B, hidden_dim)
        x = F.relu(x)
        mu = self.fc_mu(x)  # (T*B, Dz)
        logcovar = self.fc_logcovar(x)  # (T*B, Dz)
        # Reshape back to (T, B, Dx)
        mu = mu.view(B, T, self.Dz)  # (B, T, Dz)
        logcovar = logcovar.view(B, T, self.Dz)  # (B, T, Dz)

        return mu, logcovar
    
#---------------------------------------------------------------------------
#
# CNN Decoder
#
#---------------------------------------------------------------------------
#
# Takes (K,B,T,Dz) samples of z 
# decodes into (K,B,T,W=64,H=64,C=3) images
#

class DecoderCNN(nn.Module):
    """
    Decoder CNN module.
    Takes samples of latent variables of shape (K,B,T,Dz)
    and decodes them into images of shape (K,B,T,W=64, H=64, C=3).
    NB: this is the reverse of the EncoderCNN.
    """
    def __init__(self, Dz=2, hidden_dim=64):
        super(DecoderCNN, self).__init__()
        self.Dz = Dz  # Dimension of the input
        self.fc1 = nn.Linear(self.Dz, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128*8*8)
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
    
    def forward(self, x):
        """
        Input: (K,B,T,Dz)
        Output: (K,B,T,W=64,H=64,C=3)
        """
        # check input shape
        assert x.dim() == 4, f"Input shape must be (K,B,T,Dz), got {x.shape}"
        assert x.shape[3] == self.Dz, f"Input shape must be (T, B, {self.Dz}), got {x.shape}"
        
        # fwd pass
        T = x.shape[2]  # Time dimension
        B = x.shape[1]  # Batch dimension
        K = x.shape[0]  # Number of samples
        
        # Reshape to (T*B, Dx) for fully connected layers
        x = x.reshape(K*B*T, self.Dz)  # (K*B*T,Dz)
        # Fully connected layers
        x = self.fc1(x)  # (K*B*T,hidden_dim)
        x = F.relu(x)
        x = self.fc2(x)  # (K*B*T,128*8*8)
        x = F.relu(x)
        # Reshape to (K*B*T,128, 8, 8)
        x = x.view(K*B*T, 128, 8 , 8)
        # Apply transposed convolutional layers, upsampling, etc
        x = self.deconv1(x)  # (K*B*T, 64, 16, 16)
        x = F.relu(x)
        x = self.deconv2(x)  # (K*B*T, 32, 32, 32)
        x = F.relu(x)
        x = self.deconv3(x)  # (K*B*T, 3, 64, 64)
        # # Reshape back to (T, B, W=64, H=64, C=3)
        x = x.view(K,B,T,3,64,64)
        x = x.permute(0,1,2,4,5,3)  # (K,B,T,W=64,H=64,C=3)
        # output in [0,1] range
        x = torch.sigmoid(x)  # Ensure output is in [0, 1] range
        
        return x # (K,B,T,W=64,H=64,C=3)










#----------------------------------------------------------------------------------------------
# Encoder Precision : compute a precision matrix for the posterior of z
#----------------------------------------------------------------------------------------------

class EncoderPrecision(nn.Module):
    """Reprise de l'implémentation du papier GPVAE avec
    matrice de précision triangulaire supérieure à deux bandes
    """
    
    def __init__(self,
                 x_dimension = 1,
                 z_dimension = 1,
                 n_layers = 2,
                 inter_dim = 32,
                 activation = nn.ReLU,
                 epsilon = 1e-3
                 ):
        
        super(EncoderPrecision, self).__init__()
        
        self.x_dimension = int(x_dimension)
        self.z_dimension = int(z_dimension)
        self.n_layers = int(n_layers)
        self.inter_dim = int(inter_dim)
        self.activation = activation
        self.epsilon = float(epsilon) 
        
        self.diagonal_mlp = make_mlp(
            input_dim=self.x_dimension,
            output_dim=self.z_dimension,
            n_layers=self.n_layers,
            inter_dim=self.inter_dim,
            activation=activation
        )
        
        self.off_diagonal_mlp = make_mlp(
            input_dim=self.x_dimension,
            output_dim=self.z_dimension,
            n_layers=self.n_layers,
            inter_dim=self.inter_dim,
            activation=activation
        )
        
    def forward(self, x):
        """
        x : (B, N, Dx)
        return : (B, Dz, Dz)
        """
        N = x.size(-2)
        
        # Compute the diagonal part
        D = self.diagonal_mlp(x) # (B, N, Dz)
        D = torch.transpose(D, -1, -2)  # (B, Dz, N)
        D = torch.exp(D) # ensure > 0
        D = torch.diag_embed(D) # (B, Dz, N, N)
        
        # compute the upper band
        U = self.off_diagonal_mlp(x) # (B, N, Dz)
        U = torch.transpose(U, -1, -2)  # (B, Dz, N)
        U = torch.diag_embed(U[...,:-1], offset=1)  # (B, Dz, N, N)
        
        # Combine diagonal and upper band
        B = D + U # (B, Dz, N, N)

        precision_matrix = torch.transpose(B, -1, -2) @ B  # (B, N, Dz, Dz)        
        # Add epsilon to the diagonal to ensure PSD numerical stability
        epsilon_matrix = self.epsilon * torch.eye(N, device=device) # (N, N)
        precision_matrix = precision_matrix + epsilon_matrix # (B, Dz, N, N) with epsilon broacasted
        
        return D, B, precision_matrix  # (B, Dz, N, N)
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(x_dimension={self.x_dimension}, "
                f"z_dimension={self.z_dimension}, n_layers={self.n_layers}, "
                f"inter_dim={self.inter_dim}, activation={self.activation.__name__}, "
                f"epsilon={self.epsilon})")  
    
    
    
#----------------------------------------------------------------------------------------------
# Encoder Covariance : compute a precision matrix for the posterior of z
#----------------------------------------------------------------------------------------------    

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
                 epsilon = 1e-3,  # small positive constant to ensure positive definiteness of the kernel matrix
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
        self.epsilon = float(epsilon)  # small positive constant to ensure positive definiteness of the kernel matrix
        
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
        C = C + self.epsilon * torch.eye(N, device=x.device) # Add a small value to the diagonal for numerical stability (use broadcasting)
            
        return L, C # (..., Dz, N, N) Covariance matrix C
    
    def __repr__(self):
        return (f"{self.__class__.__name__} : "
                f"(x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.n_layers}, inter_dim={self.inter_dim}, "
                f"activation={self.activation.__name__}, "
                f"epsilon (to ensure PSD)={self.epsilon:.3e})")
    
    
    
    
    
    










#-------------------------------------------------------------------------------------------------
#
#--- KERNELS ------
#
#-------------------------------------------------------------------------------------------------

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
    
    def __init__(self, lengthscale=1.0, sigma=1.0, epsilon=1e-3):
        super(GaussianKernel, self).__init__()
        
        # learnable parameters for the Gaussian kernel
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))  # learnable lengthscale parameter    
        self.sigma = nn.Parameter(torch.tensor(sigma))  # learnable variance parameter
        # self.sigma = torch.tensor(sigma)  # fixed variance parameter
        
        self.epsilon = torch.tensor(epsilon, requires_grad=False)  # tolerance to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Gaussian kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (..., N) - N is typically the sequence length.
            t2 (torch.Tensor): Second set of time points (..., M)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (..., N, M)
        """
        
        assert t1.dim() == t2.dim(), "GaussianKernel object : Input tensors must have the same number of dimensions"
        
        # THE FOLLOWING TWO LINES OF CODE ARE CRITICAL TO AVOID INPLACE OPERATIONS ON THE PARAMETERS !!!!!        
        lengthscale = self.lengthscale.clone()
        sigma = self.sigma.clone()
        
        # Compute the Gaussian kernel matrix
        if t1.dim() == 1:
            t1_b = t1.unsqueeze(-1)  # (N, 1)
            t2_b = t2.unsqueeze(0)   # (1, M)
            kernel = torch.exp(-0.5 * torch.pow((t1_b - t2_b) / lengthscale, 2))  # (N, M)
        else:
            t1_b = t1.unsqueeze(-1) # (...,N, 1)
            t2_b = t2.unsqueeze(-2) # (...,1, M)
            kernel = torch.exp(-0.5 * torch.pow(torch.div(t1_b - t2_b, lengthscale),2))  # (..., N, M)
        
        gaussian_kernel_matrix = sigma**2 * kernel  # (..., N, M)

        if torch.equal(t1, t2):
            # If t1 and t2 are the same, the kernel matrix should be symmetric and positive definite
            # so we compute and return the Cholesky decomposition of the kernel matrix
            # to be used in forming the MultivariateNormal distribution, adding
            # a small value to the diagonal for numerical stability
            gaussian_kernel_matrix = gaussian_kernel_matrix + self.epsilon * torch.eye(t1.size(-1), device=t1.device, dtype=t1.dtype)

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
        return (f"{self.__class__.__name__}(lengthscale={self.lengthscale.item():.3e}, "
                f"variance (sigma**2)={self.sigma.item()**2:.3e}, "
                f"sigma={self.sigma.item():.3e}, "
                f"epsilon={self.epsilon.item():.3e})")      


class GaussianKernelFixed(nn.Module):  
    """Gaussian kernel for one univariate Gaussian Process.
    The lengthscale and variance parameters are NOT learnable.
    """
    
    # we need to add a small positive constant to ensure positive definiteness of the kernel matrix
    # 1e-3 is ok for time series up to 10,000 time points.
    # 1e-4 is ok for time series up to 2,880 time points. (ie 2 days @ 1 minute resolution).
    # 1e-4 is ok for time series up to 1,000 time points.
    # 1e-6 is ok for time series up to 100 time points.
    # the value can be decreased for shorter time series.
    # but it should not be too small, otherwise the Cholesky decomposition will fail.
    
    def __init__(self, lengthscale=1.0, sigma=1.0, epsilon=1e-3):
        super(GaussianKernelFixed, self).__init__()
        
        # learnable parameters for the Gaussian kernel
        self.lengthscale = torch.tensor(lengthscale)  # not learnable lengthscale parameter       
        self.sigma = torch.tensor(sigma)  # not learnable variance parameter
        
        self.epsilon = torch.tensor(epsilon)  # tolerance to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Gaussian kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (..., N) - N is typically the sequence length.
            t2 (torch.Tensor): Second set of time points (..., M)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (..., N, M)
        """
        
        assert t1.dim() == t2.dim(), "GaussianKernel object : Input tensors must have the same number of dimensions"
        
        # Compute the Gaussian kernel matrix
        if t1.dim() == 1:
            t1_b = t1.unsqueeze(-1)  # (N, 1)
            t2_b = t2.unsqueeze(0)   # (1, M)
            kernel = torch.exp(-0.5 * ((t1_b - t2_b) / self.lengthscale)**2)  # (N, M)
        else:
            t1_b = t1.unsqueeze(-1) # (...,N, 1)
            t2_b = t2.unsqueeze(-2) # (...,1, M)
            kernel = torch.exp(-0.5 * ((t1_b - t2_b) / self.lengthscale)**2)  # (..., N, M)
        
        gaussian_kernel_matrix = self.sigma**2 * kernel  # (..., N, M)

        if torch.equal(t1, t2):
            # If t1 and t2 are the same, the kernel matrix should be symmetric and positive definite
            # so we compute and return the Cholesky decomposition of the kernel matrix
            # to be used in forming the MultivariateNormal distribution, adding
            # a small value to the diagonal for numerical stability
            gaussian_kernel_matrix = gaussian_kernel_matrix + self.epsilon * torch.eye(t1.size(-1), device=t1.device, dtype=t1.dtype)

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
        return (f"{self.__class__.__name__}(lengthscale={self.lengthscale.item():.3e}, "
                f"variance (sigma**2)={self.sigma.item()**2:.3e}, "
                f"sigma={self.sigma.item():.3e}, "
                f"epsilon={self.epsilon.item():.3e})")        
    
    
class MaternKernel(nn.Module):
    """Matern kernel for one univariate Gaussian Process.
    Takes nu = 0.5, 1.5, 2.5 in the constructor.
    The lengthscale, variance and alpha parameters are learnable (nn.Parameter).
    """
    
    def __init__(self, nu, lengthscale=1.0, sigma=1.0, epsilon=1e-3):
        super(MaternKernel, self).__init__()
        
        # check
        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("MaternKernel: nu must be one of [0.5, 1.5, 2.5]")
        self.nu = nu  # Matern parameter nu
        
        # learnable parameters for the RQ kernel
        self.lengthscale = nn.Parameter(torch.tensor(lengthscale))  # learnable alpha parameter (shape parameter)
        self.sigma = nn.Parameter(torch.tensor(sigma))  # learnable variance = sigma**2 parameter

        self.epsilon = epsilon  # small value to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Matern kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (..., N) - N is typically the sequence length.
            t2 (torch.Tensor): Second set of time points (..., M)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (..., N, M)
        """
        
        assert t1.dim() == t2.dim(), "Matern Kernel object : Input tensors must have the same number of dimensions"
                
        # THE FOLLOWING TWO LINES OF CODE ARE CRITICAL TO AVOID INPLACE OPERATIONS ON THE PARAMETERS !!!!!        
        lengthscale = self.lengthscale.clone()
        sigma = self.sigma.clone()
        
        if t1.dim() == 1:
            t1_b = t1.unsqueeze(-1)  # (N, 1)
            t2_b = t2.unsqueeze(0)   # (1, M)
        else:
            t1_b = t1.unsqueeze(-1) # (..., N, 1)
            t2_b = t2.unsqueeze(-2) # (..., 1, M)
        
        d = torch.abs(t1_b - t2_b)  # (..., N, M)
        
        if self.nu == 0.5:
            # Matern kernel with nu = 0.5
            matern_kernel = sigma**2 * torch.exp(-d / lengthscale)
        elif self.nu == 1.5:
            # Matern kernel with nu = 1.5
            matern_kernel = sigma**2 * (1 + (d * torch.sqrt(torch.tensor(3))) / lengthscale) * torch.exp(-d * torch.sqrt(torch.tensor(3)) / lengthscale)
        elif self.nu == 2.5:
            # Matern kernel with nu = 2.5
            matern_kernel = sigma**2 * (1 + ((d * torch.sqrt(torch.tensor(5))) / lengthscale) + (5 * d**2 / (3 * lengthscale**2))) * torch.exp(-d * torch.sqrt(torch.tensor(5)) / lengthscale)
        else:
            raise ValueError("MaternKernel: nu must be one of [0.5, 1.5, 2.5]")

        if torch.equal(t1, t2):
            # If t1 and t2 are the same, the kernel matrix should be symmetric and positive definite
            # so we compute and return the Cholesky decomposition of the kernel matrix
            # to be used in forming the MultivariateNormal distribution, adding
            # a small value to the diagonal for numerical stability
            matern_kernel = matern_kernel + self.epsilon * torch.eye(t1.size(-1), device=t1.device, dtype=t1.dtype)
            try:
                L = torch.linalg.cholesky(matern_kernel)  # Cholesky decomposition to ensure positive definiteness
                return matern_kernel, L  # Return the kernel matrix and its Cholesky factor L
            except RuntimeError:
                # If the Cholesky decomposition fails, it means the matrix is not positive definite
                # We can return None or raise an error, depending on how we want to handle this case
                # print("Warning: Cholesky decomposition failed.")
                # print(f"Kernel : {gaussian_kernel_matrix}")
                # return gaussian_kernel_matrix, None  # Return the kernel matrix and None for L
                raise NameError("Cholesky decomposition of a supposedly PSD kernel matrix failed in Matern Kernel. Tolerance epsilon is likely too low.") 
        else:
            # If t1 and t2 are different, do not try to compute the Cholesky decomposition
            return matern_kernel, None
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(nu={self.nu}, "
                f"lengthscale={self.lengthscale.item():.3e}, "
                f"sigma={self.sigma.item():.3e}, "
                f"variance (sigma**2)={self.sigma.item()**2:.3e}, "
                f"epsilon={self.epsilon:.3e})")
    
    
class MaternKernelFixed(nn.Module):
    """Matern kernel for one univariate Gaussian Process.
    Takes nu = 0.5, 1.5, 2.5 in the constructor.
    The lengthscale, variance and alpha parameters are NOT learnable.
    """
    
    def __init__(self, nu, lengthscale=1.0, sigma=1.0, epsilon=1e-3):
        super(MaternKernelFixed, self).__init__()
        
        # check
        if nu not in [0.5, 1.5, 2.5]:
            raise ValueError("MaternKernel: nu must be one of [0.5, 1.5, 2.5]")
        self.nu = nu  # Matern parameter nu
        
        # learnable parameters for the RQ kernel
        self.lengthscale = torch.tensor(lengthscale)  # NOT learnable alpha parameter (shape parameter)
        self.sigma = torch.tensor(sigma)  # NOT learnable variance = sigma**2 parameter

        self.epsilon = epsilon  # small value to ensure positive definiteness of the kernel matrix
    
    def forward(self, t1, t2):
        """Compute the Matern kernel between two sets of time points.
        
        Args:
            t1 (torch.Tensor): First set of time points (..., N) - N is typically the sequence length.
            t2 (torch.Tensor): Second set of time points (..., M)
        
        Returns:
            torch.Tensor: Kernel matrix of shape (..., N, M)
        """
        
        assert t1.dim() == t2.dim(), "Matern Kernel object : Input tensors must have the same number of dimensions"
        
        if t1.dim() == 1:
            t1_b = t1.unsqueeze(-1)  # (N, 1)
            t2_b = t2.unsqueeze(0)   # (1, M)
        else:
            t1_b = t1.unsqueeze(-1) # (..., N, 1)
            t2_b = t2.unsqueeze(-2) # (..., 1, M)
        
        d = torch.abs(t1_b - t2_b)  # (..., N, M)
        
        if self.nu == 0.5:
            # Matern kernel with nu = 0.5
            matern_kernel = self.sigma**2 * torch.exp(-d / self.lengthscale)
        elif self.nu == 1.5:
            # Matern kernel with nu = 1.5
            matern_kernel = self.sigma**2 * (1 + (d * torch.sqrt(torch.tensor(3))) / self.lengthscale) * torch.exp(-d * torch.sqrt(torch.tensor(3)) / self.lengthscale)
        elif self.nu == 2.5:
            # Matern kernel with nu = 2.5
            matern_kernel = self.sigma**2 * (1 + ((d * torch.sqrt(torch.tensor(5))) / self.lengthscale) + (5 * d**2 / (3 * self.lengthscale**2))) * torch.exp(-d * torch.sqrt(torch.tensor(5)) / self.lengthscale)
        else:
            raise ValueError("MaternKernel: nu must be one of [0.5, 1.5, 2.5]")

        if torch.equal(t1, t2):
            # If t1 and t2 are the same, the kernel matrix should be symmetric and positive definite
            # so we compute and return the Cholesky decomposition of the kernel matrix
            # to be used in forming the MultivariateNormal distribution, adding
            # a small value to the diagonal for numerical stability
            matern_kernel += self.epsilon * torch.eye(t1.size(-1), device=t1.device, dtype=t1.dtype)
            try:
                L = torch.linalg.cholesky(matern_kernel)  # Cholesky decomposition to ensure positive definiteness
                return matern_kernel, L  # Return the kernel matrix and its Cholesky factor L
            except RuntimeError:
                # If the Cholesky decomposition fails, it means the matrix is not positive definite
                # We can return None or raise an error, depending on how we want to handle this case
                # print("Warning: Cholesky decomposition failed.")
                # print(f"Kernel : {gaussian_kernel_matrix}")
                # return gaussian_kernel_matrix, None  # Return the kernel matrix and None for L
                raise NameError("Cholesky decomposition of a supposedly PSD kernel matrix failed in Matern Kernel. Tolerance epsilon is likely too low.") 
        else:
            # If t1 and t2 are different, do not try to compute the Cholesky decomposition
            return matern_kernel, None
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(nu={self.nu}, "
                f"lengthscale={self.lengthscale.item():.3e}, "
                f"sigma={self.sigma.item():.3e}, "
                f"variance (sigma**2)={self.sigma.item()**2:.3e}, "
                f"epsilon={self.epsilon:.3e})")
    
    
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
    
    
class GPConstantMean(nn.Module):
    """ Neural Net to compute the mean of one univariate Gaussian Process
    This is a constant mean function, with a learnable parameter.
    """
    
    def __init__(self, constant_init=0.0):
        super(GPConstantMean, self).__init__()
        
        self.constant_value = nn.Parameter(torch.tensor(constant_init))  # learnable constant value
    
    def forward(self, t):
        """Forward pass of the GPNullMean.
        Just returns a tensor of zeros of shape identical to the input.
        
        Inputs:
            t (torch.Tensor): Input tensor of shape (..., N) - N is typically the sequence length.
        Returns:
            torch.Tensor: Output tensor of ZEROS of shape (..., N)
        """
        
        return self.constant_value * torch.ones_like(t, dtype=t.dtype, device=t.device)  # (..., N)
    
    def __repr__(self):
        return f"{self.__class__.__name__}, constant_value={self.constant_value.item():.3e})" 
    
    
    
    
#---------------------------------------------------------------------------------------
#
#--- GP PRIOR COMPUTATION
#
#---------------------------------------------------------------------------------------

def compute_gp_priors(t, Dz, kernels_list, mean_functions_list, verbose=False, tolerance=1e-3):
    """Compute the GP prior distributions for each latent dimension
    Inputs:
    - t: (B, N,) tensor of time points
    - Dz: int, dimension of the latent space
    - kernels_list: list of length Dz of kernel functions
    - mean_functions_list: list of length Dz of mean functions
    - verbose: bool, whether to print debug information
    Outputs:
    - kernel_matrix: (B, Dz, N, N) tensor of kernel matrices
    - L_matrix: (B, Dz, N, N) tensor of Cholesky factors (or None if not computed)
    - mean: (B, Dz, N) tensor of mean functions
    - kernels_list: list of length Dz of kernel functions
    - mean_functions_list: list of length Dz of mean functions
    - p_theta_z : tf.distributions.MultivariateNormal, the prior distribution over z
    """
    
    # sanity checks on dimensions
    assert len(kernels_list) == Dz, f"Length of kernels_list ({len(kernels_list)}) must be equal to Dz ({Dz})"
    assert len(mean_functions_list) == Dz, f"Length of mean_functions_list ({len(mean_functions_list)}) must be equal to Dz ({Dz})"

    # get parameters from required time stamps
    B, N = t.shape
    
    # compute the kernel matrices for each kernel
    kernel_matrices = []
    L_matrices = []
    means = []

    for i in range(Dz):
        mu_z_i = mean_functions_list[i](t)  # (B,N)
        kernel_matrix_i, L_i = kernels_list[i](t, t)  # (B,N,N), (B,N,N) or None (return Cholesky factor if possible)
        kernel_matrices.append(kernel_matrix_i)  # (B,N,N)
        L_matrices.append(L_i)  # (B,N,N) or None
        means.append(mu_z_i)  # (B,N)
        
        if verbose:
            print()
            print(f"Computing Gaussian Process prior for z component {i+1} / {Dz}...")
            print(f"Kernel {i} is : {kernels_list[i]}")
            print(f"Mean function {i} is : {mean_functions_list[i]}")
            print(f"mu_z {i} shape (B,N): {mu_z_i.shape}")  # (B, N)
            print(f"Kernel matrix {i} shape (B,N,N): {kernel_matrix_i.shape}")  # (B, N, N)
        
    kernel_matrix = torch.stack(kernel_matrices, dim=0)  # (Dz,B,N,N)
    kernel_matrix = kernel_matrix.transpose(0, 1) # (B,Dz,N,N)
    # kernel_matrix = kernel_matrix + tolerance * torch.eye(N, device=t.device).unsqueeze(0).unsqueeze(0)  # Add small value to diagonal for numerical stability (broadcasting)
    # print(f"Kernel matrix after stack and transpose (B,Dz,N,N): {kernel_matrix.shape}")  # (B,Dz,N,N)
    # print(f"Kernel matrix sample : {kernel_matrix[0,0,:,:]}")
    L_matrix = torch.stack(L_matrices, dim=0) if L_matrices[0] is not None else None  # (Dz,B,N,N) or None
    L_matrix = L_matrix.transpose(0, 1) if L_matrix is not None else None  # (B,Dz,N,N)
    mean = torch.stack(means, dim=0)  # (Dz,B,N)
    mean = mean.transpose(0, 1)  # (B,Dz,N)
    
    p_theta_z = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=kernel_matrix)
    
    if verbose:
        print()
        print(f"Full Kernel matrix shape after stack and transpose (B,Dz,N,N): {kernel_matrix.shape}")  # (B,Dz,N,N)
        print(f"Full L matrix shape after stack and transpose (B,Dz,N,N): {L_matrix.shape if L_matrix is not None else 'None'}")  # (B,Dz,N,N) or None
        print(f"Mean shape after stack and transpose (B,Dz,N): {mean.shape}")  # (B,Dz,N)
        print()
        print(f"p_theta_z batch shape : (B,Dz) = {p_theta_z.batch_shape}")  # should be (B,Dz)
        print(f"p_theta_z event shape : (N) = {p_theta_z.event_shape}")  # should be (N)
    
    return mean, kernel_matrix, L_matrix, p_theta_z
    
    
def get_parameters_from_priors(kernel_list, mean_functions_list):
    """Get the parameters of the kernels and mean functions from the lists of kernels and mean functions.
    Inputs:
    - kernel_list: list of kernel functions
    - mean_functions_list: list of mean functions
    Outputs:
    - params: list of parameters (tensors) of the kernels and mean functions
    """
    
    params = []
    
    for kernel in kernel_list:
        params += [p for p in kernel.parameters()]
        
    for mean_function in mean_functions_list:
        params += [p for p in mean_function.parameters()]
        
    return params


        
# --- brick 8 : CNN 3D Post-Decoder ----------------------------------------------
#
# --- Takes (B, T) outputs of shape (Dx)
# --- post-decodes them into (B, T, W=64, H=64, C=3)
# --- so they can be plugged as is in the VRNN model
#



#--------------------------------------------------------------------------------
#
#       TRAINING UTILITIES
#
#--------------------------------------------------------------------------------

# Early stopping class to prevent overfitting

class EarlyStoppingCallback:
    def __init__(self, patience=1, min_delta=0):
        # how many epochs do we accept with validation loss non decreasing
        self.patience = patience
        # tolerance for non decrease
        self.min_delta = min_delta
        # how many epochs without validation loss decrease
        self.counter = 0
        # minimum validation loss to beat
        self.min_validation_loss = float('inf')
        self.status = False

    def early_stop(self, validation_loss):
        # is the last validation loss better than the current minimum ?
        # status = True means stop training
        if validation_loss < self.min_validation_loss:
            # yes : update minimum value and reset counter
            self.min_validation_loss = validation_loss
            self.counter = 0
            self.status = False
            # no : are we within tolerance ?
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # no : increase counter (losing patience)
            self.counter += 1
            # have we lost patience ?
            if self.counter >= self.patience:
                # yes
                self.status = True
            else:
                # no
                self.status = False

        return self.status, self.counter    


#--------------------------------------------
# Training step function
#--------------------------------------------


def train_step(
    train_loader,
    encoder,
    decoder,
    optimizer,  
    prior,
    device,
    K, 
    beta=1.0,  # weight of the KL term in the loss 
    tolerance=1e-2  # small value to ensure positive definiteness of the kernel matrix
):
    # perform one epoch on the entire training set
    epoch_loss = 0.0
    epoch_kl = 0.0
    epoch_rec_loss = 0.0
    
    for i, data in enumerate(train_loader):
        # get data
        x = data.to(device)   # B,N,64,64,3
        # encode
        mu_x, logcovar_x = encoder(x)  # B,N,Dz x 2
        # compute q_phi (z|x)
        mu_phi = mu_x.permute(0,2,1)  # (B, Dz, N)
        covar_phi = torch.diag_embed(torch.exp(logcovar_x).permute(0,2,1))  # (B, Dz, N, N) diagonal matrices
        # covar_phi = covar_phi + tolerance * torch.eye(covar_phi.size(-1), device=covar_phi.device).unsqueeze(0).unsqueeze(0)  
        q_phi = torch.distributions.MultivariateNormal(
            loc=mu_phi, 
            covariance_matrix=covar_phi
        )

        # sample z from q_phi (z|x) using the reparameterization trick
        z_sample = q_phi.rsample((K,)) # K,B,Dz,N
        z_sample = torch.permute(z_sample,(0,1,3,2))  # K,B,N,Dz
        # compute p_theta (x|z)
        x_rec = decoder(z_sample)  # K,B,N,Dx
        
        # compute the loss
        kl = torch.distributions.kl_divergence(q_phi, prior)  # B,Dz
        kl_loss = kl.mean()  # sum over Dz
        reconstruction_loss = (x.unsqueeze(0) - x_rec)**2  # K,B,N,Dx
        # print(f"train rec shape: {reconstruction_loss.shape}, x shape: {x.shape}, x_rec shape: {x_rec.shape}")
        # reconstruction_loss = torch.nn.functional.binary_cross_entropy(x_rec, x.unsqueeze(0).expand_as(x_rec), reduction='none')  # K,B,N,Dx
        rec_loss = reconstruction_loss.mean()  # mean over B and K
        loss = rec_loss + beta * kl_loss # total loss
        
        # backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # sum losses
        epoch_loss += loss.item()
        epoch_kl += kl_loss.item()
        epoch_rec_loss += rec_loss.item()
        # report out
        print(f"Train Batch {i+1} / {len(train_loader)}: loss = {loss.item():.4e}, kl = {kl_loss.item():.4e}, rec_loss = {rec_loss.item():.4e}", end="\r")
        
    # average losses over the number of batches
    num_batches = len(train_loader)
    epoch_loss /= num_batches
    epoch_kl /= num_batches
    epoch_rec_loss /= num_batches
    
    return encoder, decoder, epoch_loss, epoch_kl, epoch_rec_loss



#--------------------------------------------
# Validation step function
#--------------------------------------------

def test_step(
    test_loader,
    encoder,
    decoder,
    prior,
    device,
    K,  
):
    # perform one epoch on the entire test set
    epoch_loss = 0.0
    epoch_kl = 0.0
    epoch_rec_loss = 0.0
    
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            # get data
            x = data.to(device)   # B,N,64,64,3
            # encode
            mu_x, logcovar_x = encoder(x)  # B,N,Dz x 2
            # compute q_phi (z|x)
            mu_phi = mu_x.permute(0,2,1)  # (B, Dz, N)
            covar_phi = torch.diag_embed(torch.exp(logcovar_x).permute(0,2,1))  # (B, Dz, N, N) diagonal matrices
            q_phi = torch.distributions.MultivariateNormal(
                loc=mu_phi, 
                covariance_matrix=covar_phi
            )

            # sample z from q_phi (z|x) using the reparameterization trick
            z_sample = q_phi.rsample((K,)) # K,B,Dz,N
            z_sample = z_sample.permute(0,1,3,2)  # K,B,N,Dz
            # compute p_theta (x|z)
            x_rec = decoder(z_sample)  # K,B,N,Dx
            
            # compute the loss
            kl = torch.distributions.kl_divergence(q_phi, prior)  # B,Dz
            kl_loss = kl.mean()  # sum over Dz
            reconstruction_loss = (x.unsqueeze(0) - x_rec)**2  # K,B,N,Dx
            # print(f"test rec shape: {reconstruction_loss.shape}, x shape: {x.shape}, x_rec shape: {x_rec.shape}")
            rec_loss = reconstruction_loss.mean()  # mean over B and K
            loss = kl_loss + rec_loss  # total loss
            
            # sum losses
            epoch_loss += loss.item()
            epoch_kl += kl_loss.item()
            epoch_rec_loss += rec_loss.item()
            # report out
            print(f"Test Batch {i+1} / {len(test_loader)}: loss = {loss.item():.4e}, kl = {kl_loss.item():.4e}, rec_loss = {rec_loss.item():.4e}", end="\r")
            
    # average losses over the number of batches
    num_batches = len(test_loader)
    epoch_loss /= num_batches
    epoch_kl /= num_batches
    epoch_rec_loss /= num_batches
    
    return epoch_loss, epoch_kl, epoch_rec_loss


#------------------------------------------------------------------
# TRAIN
#------------------------------------------------------------------

def train(
    train_loader,
    test_loader,
    encoder,
    decoder,
    optimizer,  
    prior,
    learnable_prior,
    device,
    K,  
    n_epochs=100,
):
    # log
    train_losses = []
    test_losses = []
    train_kls = []
    test_kls = []
    train_rec_losses = []
    test_rec_losses = []
    
    # train the model for n_epochs epochs
    for epoch in range(n_epochs):
        
        # PRIOR pass --------------------------------------------------------------
        # if the kernel is learnable, we need to recompute the prior mean and kernel at each epoch
        if learnable_prior:
            mean, kernel_matrix, L_matrix, prior = compute_gp_priors(t, Dz, kernels_list, mean_functions_list, verbose=False)
            
        # One train step
        encoder, decoder, train_epoch_loss, train_epoch_kl, train_epoch_rec_loss = train_step(
            train_loader,
            encoder,
            decoder,
            optimizer,  
            prior,
            device,
            K,
        )
        
        train_losses.append(train_epoch_loss)
        train_kls.append(train_epoch_kl)
        train_rec_losses.append(train_epoch_rec_loss)
        
        # One test step
        test_epoch_loss, test_epoch_kl, test_epoch_rec_loss = test_step(
            test_loader,
            encoder,
            decoder,
            prior,
            device,
            K,
        )
        
        test_losses.append(test_epoch_loss)
        test_kls.append(test_epoch_kl)
        test_rec_losses.append(test_epoch_rec_loss)
        
        # report out
        print(f"Epoch {epoch+1:<6} / {n_epochs:<6} - TRAIN : total loss {train_epoch_loss:.4e}, KL: {train_epoch_kl:.4e}, Reco: {train_epoch_rec_loss:.4e} -- TEST : total loss {test_epoch_loss:.4e}, KL: {test_epoch_kl:.4e}, Reco: {test_epoch_rec_loss:.4e}")
        
    return train_losses, test_losses, train_kls, test_kls, train_rec_losses, test_rec_losses





def report_out_losses(train_losses, test_losses, train_kls, test_kls, train_rec_losses, test_rec_losses, K):
    """ Report the losses and KL divergence.
    """
    
    # print(f"Total Loss final: {losses[-1]:.4e}")
    # print(f"KL Divergence final: {kls[-1]:.4e}")
    # print(f"Reconstruction Loss (avg over {K} sample(s)) final: {reconstruction_losses[-1]:.4e}")

    fig, ax = plt.subplots(1,3, figsize=(18, 4))

    ax[0].plot(train_losses, label='Train Loss', color='blue')
    ax[0].plot(test_losses, label='Test Loss', color='green')
    ax[0].set_title('Loss over epochs')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_yscale('log')
    ax[0].grid(True)
    ax[0].legend()

    ax[1].plot(train_kls, label='Train KL', color='blue')
    ax[1].plot(test_kls, label='Test KL', color='green')
    ax[1].set_title('KL Divergence over epochs')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('KL Divergence')
    ax[1].set_yscale('log')
    ax[1].grid(True)
    ax[1].legend()

    ax[2].plot(train_rec_losses, label='Train Rec Loss', color='blue')
    ax[2].plot(test_rec_losses, label='Test Rec Loss', color='green')
    ax[2].set_title(f'Reconstruction Loss over epochs (avg with {K} sample(s))')
    ax[2].set_xlabel('Epochs')      
    ax[2].set_ylabel('Reconstruction Loss')
    ax[2].set_yscale('log')
    ax[2].grid(True)
    ax[2].legend()

    plt.tight_layout()
    plt.show()












# ------- DEAD CODE BELOW ? ----------------------------------------------------------------

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