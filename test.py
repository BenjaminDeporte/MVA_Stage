import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from tests.unit_tests import test_brick_1, test_brick_2, test_brick_3, test_brick_4, test_brick_5, test_brick_6
from libs.dkf import DeepKalmanFilter
    
if __name__ == "__main__":
    # # Run the test for brick 1
    # print("\nTesting brick 1: Backward LSTM")
    # test_brick_1()
    
    # # Run the test for brick 2
    # print("\nTesting brick 2: Combiner")
    # test_brick_2()
    
    # # Run the test for brick 3
    # print("\nTesting brick 3: Encoder")
    # test_brick_3()
    
    # # Run the test for brick 4
    # print("\nTesting brick 4: Transition Model")
    # test_brick_4()
    
    # Run the test for brick 5
    # print("\nTesting brick 5: Decoder")
    # test_brick_5()
    
    # # Run the test for brick 6
    # print("\nTesting brick 6: Sampler")
    # test_brick_6()
    
    xdim = 1
    zdim = 8
    hdim = 16
    gdim = 2
    num_layers = 2
    layer_dim = [128, 128, 64, 32]
    
    dkf = DeepKalmanFilter(
        input_dim = xdim,
        latent_dim = zdim,
        hidden_dim = hdim,
        combiner_dim = gdim,
        num_layers = num_layers
    )
    
    seq_len = 10
    batch_size = 32
    
    # Create a random input tensor
    x = torch.randn(seq_len, batch_size, xdim)
    
    # test dkf
    x, mu_x_s, logvar_x_s, mu_z_s, logvar_z_s, mu_z_transition_s, logvar_z_transition_s = dkf(x)
    