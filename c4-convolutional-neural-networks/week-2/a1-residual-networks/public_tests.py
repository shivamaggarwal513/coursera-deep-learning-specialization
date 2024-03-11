import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor

from dlai_tools.testing_utils import comparator, summary
from termcolor import colored
from outputs import *


def identity_block_test(target):
    tf.random.set_seed(2)
    np.random.seed(1)
    
    #X = np.random.randn(3, 4, 4, 6).astype(np.float32)
    X1 = np.ones((1, 4, 4, 3)) * -1
    X2 = np.ones((1, 4, 4, 3)) * 1
    X3 = np.ones((1, 4, 4, 3)) * 3

    X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)
    
    tf.keras.backend.set_learning_phase(False)
    A3 = target(X, f = 2, filters = [4, 4, 3], initializer=lambda seed=0:tf.keras.initializers.constant(value=1))
    A3np = A3.numpy()
    
    assert tuple(A3np.shape) == (3, 4, 4, 3), "Shapes does not match. This is really weird"
    assert np.all(A3np >= 0), "The ReLu activation at the last layer is missing"
    resume = A3np[:,(0,-1),:,:].mean(axis = 3)

    assert np.floor(resume[1, 0, 0]) == 2 * np.floor(resume[1, 0, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 0, 3]) == np.floor(resume[1, 1, 0]),     "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"
    assert np.floor(resume[1, 1, 0]) == 2 * np.floor(resume[1, 1, 3]), "Check the padding and strides"

    assert resume[1, 1, 0] - np.floor(resume[1, 1, 0]) > 0.7, "Looks like the BatchNormalization units are not working"
    
    assert np.allclose(resume, identity_block_output1, atol = 1e-5 ), "Wrong values with training=False"

    tf.keras.backend.set_learning_phase(True)
    np.random.seed(1)
    tf.random.set_seed(2)
    A4 = target(X, f = 3, filters = [3, 3, 3], initializer=lambda seed=7:tf.keras.initializers.constant(value=1))
    A4np = A4.numpy()
    resume = A4np[:,(0,-1),:,:].mean(axis = 3)
    
    assert np.allclose(resume, identity_block_output2, atol = 1e-5 ), "Wrong values with training=True"

    print(colored("All tests passed!", "green"))

    
def convolutional_block_test(target):
    np.random.seed(1)
    tf.random.set_seed(2)
    
    #X = np.random.randn(3, 4, 4, 6).astype(np.float32)
    X1 = np.ones((1, 4, 4, 3)) * -1
    X2 = np.ones((1, 4, 4, 3)) * 1
    X3 = np.ones((1, 4, 4, 3)) * 3

    X = np.concatenate((X1, X2, X3), axis = 0).astype(np.float32)

    tf.keras.backend.set_learning_phase(False)
    
    A = target(X, f = 2, s = 4, filters = [2, 4, 6])
    assert tuple(tf.shape(A).numpy()) == (3, 1, 1, 6), "Wrong shape. Make sure you are using the stride values as expected."
    
    B = target(X, f = 2, filters = [2, 4, 6])
    assert type(B) == EagerTensor, "Use only tensorflow and keras functions"
    assert tuple(tf.shape(B).numpy()) == (3, 2, 2, 6), "Wrong shape."
    assert np.allclose(B.numpy(), convolutional_block_output1), "Wrong values when training=False."
    print(B[0])
    
    tf.keras.backend.set_learning_phase(True)
    
    C = target(X, f = 2, filters = [2, 4, 6])
    assert np.allclose(C.numpy(), convolutional_block_output2), "Wrong values when training=True."

    print(colored("All tests passed!", "green"))


def ResNet50_test(target):
    comparator(summary(target(input_shape = (64, 64, 3), classes = 6)), ResNet50_summary)
