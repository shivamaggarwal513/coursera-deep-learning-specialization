import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from termcolor import colored
from fr_utils import model, database


def triplet_loss_test(target):
    tf.random.set_seed(1)
    
    y_true = (None, None, None) # It is not used
    y_pred = (tf.random.normal([3, 128], mean=6, stddev=0.1, seed = 1),
              tf.random.normal([3, 128], mean=1, stddev=1, seed = 1),
              tf.random.normal([3, 128], mean=3, stddev=4, seed = 1))
    loss = target(y_true, y_pred)

    assert type(loss) == EagerTensor, "Use tensorflow functions"
    print("loss = " + str(loss))

    y_pred_perfect = ([[1., 1.]], [[1., 1.]], [[1., 1.,]])
    loss = target(y_true, y_pred_perfect, 5)
    assert loss == 5, "Wrong value. Did you add the alpha to basic_loss?"
    
    y_pred_perfect = ([[1., 1.]],[[1., 1.]], [[0., 0.,]])
    loss = target(y_true, y_pred_perfect, 3)
    assert loss == 1., "Wrong value. Check that pos_dist = 0 and neg_dist = 2 in this example"
    
    y_pred_perfect = ([[1., 1.]],[[0., 0.]], [[1., 1.,]])
    loss = target(y_true, y_pred_perfect, 0)
    assert loss == 2., "Wrong value. Check that pos_dist = 2 and neg_dist = 0 in this example"
    
    y_pred_perfect = ([[0., 0.]],[[0., 0.]], [[0., 0.,]])
    loss = target(y_true, y_pred_perfect, -2)
    assert loss == 0, "Wrong value. Are you taking the maximum between basic_loss and 0?"
    
    y_pred_perfect = ([[1., 0.], [1., 0.]],[[1., 0.], [1., 0.]], [[0., 1.], [0., 1.]])
    loss = target(y_true, y_pred_perfect, 3)
    assert loss == 2., "Wrong value. Are you applying tf.reduce_sum to get the loss?"
    
    y_pred_perfect = ([[1., 1.], [2., 0.]], [[0., 3.], [1., 1.]], [[1., 0.], [0., 1.,]])
    loss = target(y_true, y_pred_perfect, 1)
    assert loss != 4.,"Perhaps you are not using axis=-1 in reduce_sum?"
    assert loss == 5, "Wrong value. Check your implementation"
    
    print(colored("All tests passed!", "green"))


def verify_test(target):
    distance, door_open_flag = target("faces/younes_1.jpg", "younes", database, model)
    assert np.isclose(distance, 0.60197896), "Distance not as expected"
    assert isinstance(door_open_flag, bool), "Door open flag should be a boolean"
    assert door_open_flag == True, "Door open flag not as expected, check distance threshold"
    
    distance, door_open_flag = target("faces/benoit_1.jpg", "kian", database, model)
    assert np.isclose(distance, 1.0130049), "Distance not as expected"
    assert isinstance(door_open_flag, bool), "Door open flag should be a boolean"
    assert door_open_flag == False, "Door open flag not as expected, check distance threshold"
    
    print(colored("All tests passed!", "green"))


def who_is_it_test(target):
    test1 = target("faces/younes_1.jpg", database, model)
    assert np.isclose(test1[0], 0.60197896)
    assert test1[1] == "younes"
    
    test2 = target("faces/bertrand_2.jpg", database, model)
    assert np.isclose(test2[0], 0.39049482)
    assert test2[1] == "bertrand"
    
    print(colored("All tests passed!", "green"))
