import os

from tests.unit_tests import test_brick_1, test_brick_2, test_brick_3
    
if __name__ == "__main__":
    # Run the test for brick 1
    print("\nTesting brick 1: Backward LSTM")
    test_brick_1()
    
    # Run the test for brick 2
    print("\nTesting brick 2: Combiner")
    test_brick_2()
    
    # Run the test for brick 3
    print("\nTesting brick 3: Encoder")
    test_brick_3()
    