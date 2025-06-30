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
    
    
class ShapeManager():
    """Utility class to manage the shape of tensors.
    Takes a tensor as input, learns it shape : (B x L x D)
    provides a method to reshape from (L x D) or (D) to (B x L x D).
    provides a method to reshape from (B x L x D) to the original (L x D) or (D).
    """
    
    def __init__(self, x):
        """
        Initialize the ShapeManager with a tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D) or (L, D) or (D).
            if x is (B, L, D), then
                - shape_code = 3, batch_size = B, sequence_length = L
            if x is (L, D), then
                - shape_code = 2, batch_size = 1, sequence_length = L
            if x is (D), then
                - shape_code = 1, batch_size = 1, sequence_length = 1
        """
        
        self.number_dim = x.dim()
        
        if x.dim() == 3:
            self.batch_size = x.size(0)
            self.sequence_length = x.size(1)
            self.x_dimension = x.size(2)
            self.shape_code = 3
        elif x.dim() == 2:
            self.batch_size = 1
            self.sequence_length = x.size(0)
            self.x_dimension = x.size(1)
            self.shape_code = 2
        elif x.dim() == 1:
            self.batch_size = 1
            self.sequence_length = 1
            self.x_dimension = x.size(0)
            self.shape_code = 1
        else:
            raise ValueError("Error in Shape Manager : input tensor must have 1, 2, or 3 dimensions.")
        
    def shape_in(self, input):
        """
        Takes an input tensor, cast it to shape (B, L, D).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D) or (L, D) or (D).
            must be same shape as the one used to initialize the ShapeManager.
        
        Returns:
            torch.Tensor: Reshaped tensor of shape (B, L, D).
            - (D) => (1, 1, D)
            - (L, D) => (1, L, D)
            - (B, L, D) => (B, L, D)
        """
        
        assert input.dim() in [1, 2, 3], "Input tensor must have 1, 2, or 3 dimensions."
        # assert input.dim() == self.number_dim, "Input tensor must have the same number of dimensions as the one used to initialize the ShapeManager."
        
        if input.dim() == 1:
            # input is (D)
            return input.unsqueeze(0).unsqueeze(0)
        elif input.dim() == 2:
            # input is (L, D)
            return input.unsqueeze(0)
        elif input.dim() == 3:
            # input is (B, L, D)
            return input
        else:
            raise ValueError("Error in Shape Manager : input tensor must have 1, 2, or 3 dimensions.")
        
    def shape_out(self, output):
        """
        Takes an output tensor (B, L, D), recast it to the original shape given at initialization.
        """
        
        if self.shape_code == 3:
            # output is (B, L, D)
            return output.view(self.batch_size, self.sequence_length, -1)
        elif self.shape_code == 2:
            # output is (L, D)
            return output.view(self.sequence_length, -1)
        elif self.shape_code == 1:
            # output is (D)
            return output.view(-1)
        else:
            raise ValueError("Error in Shape Manager : original_dim must be 1, 2, or 3.")
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(shape_code={self.shape_code}, "
                f"batch_size={self.batch_size}, sequence_length={self.sequence_length}, "
                f"x_dimension={self.x_dimension})")
    
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
            x (torch.Tensor): Input tensor of either shape :
                - (batch_size, sequence_length, x_dimension)
                - (sequence_length, x_dimension). 
                - (x_dimension)
        Returns:
            torch.Tensor: Output tensor of shape corresponding to the input shape
                - (batch_size, sequence_length, z_dimension)
                - (sequence_length, z_dimension)
                - (z_dimension)
                
        NB : z_dimension = 1 in this first implementation.
        """
        
        sm = ShapeManager(x)
        x = sm.shape_in(x) # (B, L, Dx)
        x = x.view(x.size(0), self.sequence_length * self.x_dimension)  # Flatten the input tensor (B, L*Dx)
        x = self.mlp(x) # (B, L) (remember z_dimension = 1)
        x = sm.shape_out(x)  # Reshape to the original shape
            
        return x
    
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
        
        # NN to compute the elements of the diagonal. 
        # There are sequence_length * z_dimension elements in the diagonal of the covariance matrix.
        # Outputs must be strictly positive. This is done in the forward pass by applying the exponential function.
 
        # # form the list of layers
        # layers = []
        # for i in range(n_layers):
        #     if i == 0:
        #         input_dim = sequence_length * x_dimension
        #     else:
        #         input_dim = inter_dim
                
        #     if i == n_layers - 1:
        #         output_dim = sequence_length
        #     else:
        #         output_dim = inter_dim
                
        #     layers.append(nn.Linear(input_dim, output_dim))
        #     if i < n_layers - 1:
        #         layers.append(self.activation())
                
        self.diagonal_mlp = make_mlp(
            input_dim=sequence_length * x_dimension,
            output_dim=sequence_length,  # output is a vector of length sequence_length
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
                       
        # self.diagonal_mlp = nn.Sequential(*layers) # IN : (B or 1) x (T * Dx) => OUT: (B or 1) x T (nb Dz = 1)
        
        # NN to compute the elements of the lower triangular part of the covariance matrix.
        # The covariance matrix is of shape (sequence_length * z_dimension, sequence_length * z_dimension). (here z_dimension = 1)
        # We actually compute a full matrix of shape (sequence_length * z_dim) x (sequence_length * z_dim), but we will only use the lower triangular part.
        
        # layers = []
        # for i in range(self.n_layers):
        #     if i == 0:
        #         input_dim = self.sequence_length * self.x_dimension
        #     else:
        #         input_dim = self.inter_dim
                
        #     if i == self.n_layers - 1:
        #         output_dim = self.sequence_length * self.sequence_length
        #     else:
        #         output_dim = self.inter_dim
                
        #     layers.append(nn.Linear(input_dim, output_dim))
        #     if i < n_layers - 1:
        #         layers.append(self.activation())
                
        self.full_matrix_mlp = make_mlp(
            input_dim=self.sequence_length * self.x_dimension,
            output_dim=self.sequence_length * self.sequence_length,  # output is a full matrix of shape (T*T)
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
        
        # self.full_matrix_mlp = nn.Sequential(*layers) # IN : (B or 1) x (T * Dx) => OUT: (B or 1) x (T*T)
    
    def forward(self, x):
        """
        Forward pass of the EncoderCovariance.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, x_dimension) or (sequence_length, x_dimension). 
            If not batch_dimension is specified, add batch_size=1 for the forward pass, and remove the batch_dimension in the output.
        Returns:
            L (torch.Tensor): Lower triangular matrix of shape (batch_size, sequence_length * z_dimension, sequence_length * z_dimension) 
                or (sequence_length * z_dimension, sequence_length * z_dimension) if no batch specified.
                Here : z_dimension = 1.
            C (torch.Tensor): Covariance matrix of shape (batch_size, sequence_length * z_dimension, sequence_length * z_dimension) 
                or (sequence_length * z_dimension, sequence_length * z_dimension) if no batch specified.
                Here : z_dimension = 1.
            C is computed as L @ L^T, where L is the lower triangular matrix. (Cholesky decomposition)
        """
        
        # manages shape of input x
        batch_specified = (x.dim() == 3)
        if batch_specified:
            batch_size = x.size(0)
        else:
            batch_size = 1
            x = x.unsqueeze(0)
        # here, x has shape (B or 1, sequence_length, x_dimension)
            
        assert x.size(1) == self.sequence_length, f"Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        assert x.size(2) == self.x_dimension, f"Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        
        x = x.view(x.size(0), -1)  # Flatten the input tensor => x has shape B x (L * Dx)
        
        # Compute the diagonal elements of the covariance matrix
        D = self.diagonal_mlp(x)  # out : (B or 1) x T (Dz=1)
        D = torch.exp(self.diagonal_mlp(x)) # out : (B or 1) x T. Ensure strictly positive diagonal elements.
        D = torch.diag_embed(D) # shape (B or 1) x T x T
        
        # Get the elements outside the diagonal
        M = self.full_matrix_mlp(x) # shape (B or 1) x (T*T)
        M = M.reshape(batch_size, self.sequence_length * self.z_dimension, self.sequence_length * self.z_dimension) # M is a full matrix of shape (B or 1) x T x T
        idx = torch.arange(self.sequence_length * self.z_dimension, device=x.device)  # Create an index tensor for the diagonal elements
        M[:,idx,idx] = 0.0  # Fill the diagonal with 0s
        
        # Assemble the lower triangular matrix L
        L = torch.zeros_like(M, device=x.device)  # Initialize L with zeros
        L = D + M
        L[:] = torch.tril(L[:])
        
        if not batch_specified:
            L = L.view(self.sequence_length * self.z_dimension, self.sequence_length * self.z_dimension)
            
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
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, x_dimension) or (sequence_length, x_dimension). 
            If no batch dimension is specified, then add batch_size=1 for the forward pass, and remove the batch_dimension in the output.
        Returns:
            mu (torch.Tensor): Mean of the approximate posterior distribution of shape (batch_size, sequence_length) or (sequence_length) if no batch specified.
            Sigma (torch.Tensor): Covariance matrix of the approximate posterior distribution of shape (batch_size, sequence_length, sequence_length) or (sequence_length, sequence_length) if no batch specified.
            q_phi (torch.distributions.MultivariateNormal): Multivariate normal distribution with parameters mu and Sigma.
        """
        
        # manage shape of x
        sm = ShapeManager(x)
        x = sm.shape_in(x)  # (B, L, Dx)
        
        # batch_specified = (x.dim() == 3)
        # if batch_specified:
        #     batch_size = x.size(0)
        # else:
        #     # if x is (T x Dx), add a batch dimension to (B x T x Dx)
        #     batch_size = 1
        #     x = x.unsqueeze(0)
        batch_size = x.size(0) 
        sequence_length = x.size(1)
        x_dim = x.size(2)
        
        assert sequence_length == self.sequence_length, f"Incorrect sequence length passed to Encoder. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        assert x_dim == self.x_dimension, f"Incorrect x_dimension passed to Encoder. Input tensor must have shape (batch_size, {self.sequence_length}, {self.x_dimension}) or ({self.sequence_length}, {self.x_dimension})"
        
        # compute parameters of the approximate posterior distribution
        mu = self.encoder_mean(x).squeeze()  # shape (batch_size, sequence_length) or (sequence_length) - assume z_dim = 1 
        _, sigma = self.encoder_covariance(x)  # shape (batch_size, sequence_length, sequence_length) or (sequence_length, sequence_length)
        
        # instantiate the multivariate normal distribution
        q_phi = torch.distributions.MultivariateNormal(mu, sigma)
        
        return mu, sigma, q_phi        
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(sequence_length={self.sequence_length}, "
                f"x_dimension={self.x_dimension}, z_dimension={self.z_dimension}, "
                f"n_layers={self.encoder_mean.n_layers}, inter_dim={self.encoder_mean.inter_dim}, "
                f"activation={self.encoder_mean.activation.__name__})")


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
        
        # # form the list of layers
        # layers = []
        # for i in range(n_layers):
        #     if i == 0:
        #         input_dim = sequence_length * x_dimension
        #     else:
        #         input_dim = inter_dim
                
        #     if i == n_layers - 1:
        #         output_dim = sequence_length
        #     else:
        #         output_dim = inter_dim
        
        #     layers.append(nn.Linear(input_dim, output_dim))
        #     if i < n_layers - 1:
        #         layers.append(self.activation())
        
        # self.mlp = nn.Sequential(*layers)
        
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
            z (torch.Tensor): Input tensor of shape (batch_size, sequence_length, z_dimension) or (sequence_length, z_dimension) or (z_dimension,).
            The output shape matches the input shape.
        Returns:
            torch.Tensor: mu_x : Output tensor of shape (batch_size, sequence_length, x_dimension) or (sequence_length, x_dimension) or (x_dimension,).
        """
        
        # manages input shape
        # if z.dim() == 1:
        #     z_dim = z.size(0)
        #     z = z.unsqueeze(0, 1)  # Add batch and length dimensions
        #     batch_size = 1
        #     sequence_length = 1
        #     batch_specified = False
        #     length_specified = False
        # elif z.dim() == 2:
        #     z_dim = z.size(1)
        #     sequence_length = z.size(0)
        #     batch_specified = False
        #     length_specified = True
        #     z = z.unsqueeze(0)  # Add batch dimension
        #     batch_size = 1
        # elif z.dim() == 3:
        #     z_dim = z.size(2)
        #     sequence_length = z.size(1)
        #     batch_size = z.size(0)
        #     batch_specified = True
        #     length_specified = True
        # else:
        #     raise ValueError("Input tensor z must have 1, 2, or 3 dimensions.")
        
        # maange execution depending on the shape of z
        if z.dim() == 1:
            z = z.unsqueeze(0).unsqueeze(0)  # Add batch and length dimensions (1, 1, Dz)
            batch_size = 1
            batch_specified = False
            sequence_length = 1
            lenght_specified = False
        elif z.dim() == 2:
            z = z.unsqueeze(0)  # Add batch dimension (1, L, Dz)
            batch_size = 1
            batch_specified = False
            sequence_length = z.size(0)
            legnth_specified = True
        elif z.dim() == 3:
            batch_size = z.size(0)
            batch_specified = True
            sequence_length = z.size(1)
            length_specified = True
        else:
            raise ValueError("Input tensor z must have 1, 2, or 3 dimensions.")
        
        # mini check
        z_dim = z.size(-1)
        assert z_dim == self.z_dimension, f"Incorrect latent dim. Input tensor z must have shape (batch_size, sequence_length, {self.z_dimension}) or ({self.z_dimension},) or ({self.z_dimension}, sequence_length) or (sequence_length, {self.z_dimension})"
        
        # compute stuff
        mu_x = self.mlp(z)  # Apply the MLP - mu_x (1, 1, Dx)

        
                   
        z = z.view(z.size(0), -1)  # Flatten the input tensor (B, L*Dz)
        mu_x = self.mlp(z)  # Apply the MLP - mu_x (B, L * Dx)
        mu_x = sm.shape_out(mu_x)  # Reshape to the original shape (B, L, Dx) or (L, Dx) or (Dx,)
        # mu_x = mu_x.view(batch_size, sequence_length, self.x_dimension)  # Reshape to (batch_size, sequence_length, z_dimension)
        
        # # gives back the original shape
        # if not batch_specified and not length_specified:
        #     mu_x = mu_x.squeeze(0, 1)
        # if not batch_specified and length_specified:
        #     mu_x = mu_x.squeeze(0)
            
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
        
        # form the list of layers
        # layers = []
        # for i in range(n_layers):
        #     if i == 0:
        #         input_dim = sequence_length * x_dimension
        #     else:
        #         input_dim = inter_dim
                
        #     if i == n_layers - 1:
        #         output_dim = sequence_length
        #     else:
        #         output_dim = inter_dim
        
        #     layers.append(nn.Linear(input_dim, output_dim))
        #     if i < n_layers - 1:
        #         layers.append(self.activation())
        
        # self.mlp = nn.Sequential(*layers)
        
        self.mlp = make_mlp(
            input_dim=sequence_length * z_dimension,
            output_dim=sequence_length,  # output is a vector of length sequence_length
            n_layers=n_layers,
            inter_dim=inter_dim,
            activation=activation
        )
    
    def forward(self, z):
        """
        Forward pass of the DecoderCovariance.
        Args:
            z (torch.Tensor): Input tensor of shape (batch_size, sequence_length, z_dimension) or (sequence_length, z_dimension) or (z_dimension,).
            The output shape matches the input shape.
        Returns:
            torch.Tensor: Logvar - Output tensor of shape (batch_size, sequence_length, x_dimension) or (sequence_length, x_dimension) or (x_dimension,).
        """
        
        # # manages input shape of z²
        # if z.dim() == 1:
        #     z_dim = z.size(0)
        #     z = z.unsqueeze(0, 1)  # Add batch and length dimensions
        #     batch_size = 1
        #     sequence_length = 1
        #     batch_specified = False
        #     length_specified = False
        # elif z.dim() == 2:
        #     z_dim = z.size(1)
        #     sequence_length = z.size(0)
        #     batch_specified = False
        #     length_specified = True
        #     z = z.unsqueeze(0)  # Add batch dimension
        #     batch_size = 1
        # elif z.dim() == 3:
        #     z_dim = z.size(2)
        #     sequence_length = z.size(1)
        #     batch_size = z.size(0)
        #     batch_specified = True
        #     length_specified = True
        # else:
        #     raise ValueError("Input tensor z must have 1, 2, or 3 dimensions.")
        
        sm = ShapeManager(z)
        z = sm.shape_in(z)  # (B, L, Dz)
        z_dim = z.size(2)
        
        assert z_dim == self.z_dimension, f"Incorrect latent dim. Input tensor z must have shape (batch_size, sequence_length, {self.z_dimension}) or ({self.z_dimension},) or ({self.z_dimension}, sequence_length) or (sequence_length, {self.z_dimension})"
                   
        z = z.view(z.size(0), -1)  # Flatten the input tensor (B, L*Dz)
        logvar_x = self.mlp(z)  # Apply the MLP
        logvar_x = sm.shape_out(logvar_x)  # Reshape to the original shape (B, L, Dx) or (L, Dx) or (Dx,)
        # logvar_x = logvar_x.view(batch_size, sequence_length, self.x_dimension)  # Reshape to (batch_size, sequence_length, z_dimension)
        
        # # gives back the original shape
        # if not batch_specified and not length_specified:
        #     logvar_x = logvar_x.squeeze(0, 1)
        # if not batch_specified and length_specified:
        #     logvar_x = logvar_x.squeeze(0)
            
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
                    activation = nn.ReLU,
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
        """Takes a sequence of length N of latent variables z_{1:N}
        computes the parameters of the observation model p_{\theta_x}(x_{1:N}|z_{1:N}):
            - mu_x(z_{1:N}): mean of the observation model
            - logvar_x(z_{1:N}): log variance of the observation model
            - p_theta_x: the observation model itself.
        NB : N can be 1, T or another strictly positive integer.
        """
        
        # manage shape of z
        sm = ShapeManager(z)
        z = sm.shape_in(z)  # (B, N, Dz)
        
        mu_x = self.decoder_mean(z)  # (B, N, Dx)) or (N, Dx) or (Dx)
        logvar_x = self.decoder_covariance(z)  # (B, N, Dx) or (N, Dx) or (Dx)

        # instantiate the multivariate normal distribution
        p_theta_x = torch.distributions.MultivariateNormal(mu_x, torch.diag_embed(torch.exp(logvar_x)))
        
        return mu_x, logvar_x, p_theta_x





#---------------------------------------------------------------------
#----------------------------------------------------------------------
#
# TESTS
#
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
    
    # TEST SHAPE MANAGER
    print("\nTesting Shape Manager...")
    
    # ----------------
    print("\nTest 0 : instantiation with (B,L,D)")
    B, L, D = 16, 10, 5
    x = torch.randn(B, L, D)
    sm = ShapeManager(x)
    print(f"Instantiation shape: {x.shape}")
    print(sm)
    
    print(f"Testing cast from (B,L,D), (L,D), (D) to (B,L,D)")
    input = torch.randn(B, L, D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(L, D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    
    print(f"Testing cast from (D), (L,D), (B,L,D) to original shape (B,L,D)")
    input = torch.randn(B*L*3)
    output = sm.shape_out(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(B*L, D)
    output = sm.shape_out(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(B, L, D)
    output = sm.shape_out(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    
    #----------------------
    print("\nTest 1 : instantiation with (L,D)")
    L, D = 50, 3
    x = torch.randn(L, D)
    sm = ShapeManager(x)
    print(f"Instantiation shape: {x.shape}")
    print(sm)
    
    print(f"Testing cast from (B,L,D), (L,D), (D) to (B,L,D)")
    input = torch.randn(B, L, D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(L, D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    
    print(f"Testing cast from (D), (L,D), (B,L,D) to original shape (B,L,D)")
    input = torch.randn(D*L)
    output = sm.shape_out(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(L, D)
    output = sm.shape_out(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    # input = torch.randn(B, L, D)
    # output = sm.shape_out(input)
    # print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    
    #----------------------
    print("\nTest 2 : instantiation with (D)")
    D = 32
    x = torch.randn(D)
    sm = ShapeManager(x)
    print(f"Instantiation shape: {x.shape}")
    print(sm)
    
    print(f"Testing cast from (B,L,D), (L,D), (D) to (B,L,D)")
    input = torch.randn(B, L, D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(L, D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    input = torch.randn(D)
    output = sm.shape_in(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    
    print(f"Testing cast from (D), (L,D), (B,L,D) to original shape (B,L,D)")
    input = torch.randn(D)
    output = sm.shape_out(input)
    print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    # input = torch.randn(B*L, D)
    # output = sm.shape_out(input)
    # print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    # input = torch.randn(B, L, D)
    # output = sm.shape_out(input)
    # print(f"Input shape: {input.shape} => Output shape: {output.shape}")
    
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
    
    # test 1
    print("\nTest Encoder 1 : forward pass with batch dimension")
    x = torch.randn(16, sequence_length, x_dimension)  # batch_size=2
    print(f"Input shape: {x.shape}")
    mu, sigma, q_phi = encoder(x)
    print(f"Output mu shape: {mu.shape}")
    print(f"Output sigma shape: {sigma.shape}")
    print(f"Output q_phi: {q_phi}")
    
    # test 2
    print("\nTest Encoder 2 : forward pass without batch dimension")
    x = torch.randn(sequence_length, x_dimension)  # no batch dimension
    print(f"Input shape: {x.shape}")
    mu, sigma, q_phi = encoder(x)
    print(f"Output mu shape: {mu.shape}")
    print(f"Output sigma shape: {sigma.shape}")
    print(f"Output q_phi: {q_phi}")
    
    # DECODER TESTS
    print("Test Decoder 0 : instantiation...")
    
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
    
    print(f"\nTest Decoder 2 : forward pass without batch dimension")
    z = torch.randn(sequence_length, z_dimension)  # no batch dimension
    print(f"Input shape: {z.shape}")
    mu_x, logvar_x, p_theta_x = decoder(z)
    print(f"Output mu_x shape: {mu_x.shape}")
    print(f"Output logvar_x shape: {logvar_x.shape}")
    print(f"Output p_theta_x: {p_theta_x}")
    
    print(f"\nTest Decoder 3 : forward pass with a single latent variable z_t")
    z = torch.randn(z_dimension)  # single latent variable
    print(f"Input shape: {z.shape}")
    mu_x, logvar_x, p_theta_x = decoder(z)
    print(f"Output mu_x shape: {mu_x.shape}")
    print(f"Output logvar_x shape: {logvar_x.shape}")
    print(f"Output p_theta_x: {p_theta_x}")