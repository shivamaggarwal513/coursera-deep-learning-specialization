import tensorflow as tf
from dlai_tools.testing_utils import summary, comparator
from outputs import *


def conv_block_test(target):
    input_size = (96, 128, 3)
    n_filters  = 32
    inputs     = tf.keras.layers.Input(input_size)
    cblock1    = target(inputs, n_filters * 1)
    model1     = tf.keras.Model(inputs=inputs, outputs=cblock1)
    
    summary1   = summary(model1)

    print('Block 1:')
    for layer in summary1:
        print(layer)

    comparator(summary1, conv_block_output1)

    inputs   = tf.keras.layers.Input(input_size)
    cblock1  = target(inputs, n_filters * 32, dropout_prob=0.1, max_pooling=True)
    model2   = tf.keras.Model(inputs=inputs, outputs=cblock1)
    
    summary2 = summary(model2)
            
    print('\nBlock 2:')   
    for layer in summary2:
        print(layer)
        
    comparator(summary2, conv_block_output2)


def upsampling_block_test(target):
    input_size1        = (12, 16, 256)
    input_size2        = (24, 32, 128)
    n_filters          = 32
    expansive_inputs   = tf.keras.layers.Input(input_size1)
    contractive_inputs = tf.keras.layers.Input(input_size2)
    cblock1            = target(expansive_inputs, contractive_inputs, n_filters)
    model1             = tf.keras.Model(inputs=[expansive_inputs, contractive_inputs], outputs=cblock1)
    
    summary1 = summary(model1)
    
    print('Block 1:')
    for layer in summary1:
        print(layer)

    comparator(summary1, upsampling_block_output)


def unet_model_test(target):
    img_height   = 96
    img_width    = 128
    num_channels = 3
    unet         = target((img_height, img_width, num_channels))
    
    comparator(summary(unet), unet_model_output)
