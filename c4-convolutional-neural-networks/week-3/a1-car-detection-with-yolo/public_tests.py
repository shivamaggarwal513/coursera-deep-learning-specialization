import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


def yolo_filter_boxes_test(target):
    tf.random.set_seed(10)
    box_confidence = tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1)
    boxes = tf.random.normal([19, 19, 5, 4], mean=1, stddev=4, seed = 1)
    box_class_probs = tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = target(boxes, box_confidence, box_class_probs, threshold = 0.5)

    assert type(scores) == EagerTensor, "Use tensorflow functions"
    assert type(boxes) == EagerTensor, "Use tensorflow functions"
    assert type(classes) == EagerTensor, "Use tensorflow functions"

    assert scores.shape == (1789,), "Wrong shape in scores"
    assert boxes.shape == (1789, 4), "Wrong shape in boxes"
    assert classes.shape == (1789,), "Wrong shape in classes"

    assert np.isclose(scores[2].numpy(), 9.270486), "Values are wrong on scores"
    assert np.allclose(boxes[2].numpy(), [4.6399336, 3.2303846, 4.431282, -2.202031]), "Values are wrong on boxes"
    assert classes[2].numpy() == 8, "Values are wrong on classes"

    print("\033[92m All tests passed!")


def iou_test(target):
    ## Test case 1: boxes intersect
    box1 = (2,1,4,3)
    box2 = (1,2,3,4)
    assert target(box1, box2) < 1, "The intersection area must be always smaller or equal than the union area."
    assert np.isclose(target(box1, box2), 0.14285714), "Wrong value. Check your implementation. Problem with intersecting boxes"

    ## Test case 2: boxes do not intersect
    box1 = (1,2,3,4)
    box2 = (5,6,7,8)
    assert target(box1, box2) == 0, "Intersection must be 0"

    ## Test case 3: boxes intersect at vertices only
    box1 = (1,1,2,2)
    box2 = (2,2,3,3)
    assert target(box1, box2) == 0, "Intersection at vertices must be 0"

    ## Test case 4: boxes intersect at edge only
    box1 = (1,1,3,3)
    box2 = (2,3,3,4)
    assert target(box1, box2) == 0, "Intersection at edges must be 0"

    print("\033[92m All tests passed!")


def yolo_non_max_suppression_test(target):
    tf.random.set_seed(10)
    scores = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
    boxes = tf.random.normal([54, 4], mean=1, stddev=4, seed = 1)
    classes = tf.random.normal([54,], mean=1, stddev=4, seed = 1)
    scores, boxes, classes = target(scores, boxes, classes)

    assert type(scores) == EagerTensor, "Use tensoflow functions"
    assert type(boxes) == EagerTensor, "Use tensoflow functions"
    assert type(classes) == EagerTensor, "Use tensoflow functions"

    assert scores.shape == (10,), "Wrong shape"
    assert boxes.shape == (10, 4), "Wrong shape"
    assert classes.shape == (10,), "Wrong shape"

    assert np.isclose(scores[2].numpy(), 8.147684), "Wrong value on scores"
    assert np.allclose(boxes[2].numpy(), [ 6.0797963, 3.743308, 1.3914018, -0.34089637]), "Wrong value on boxes"
    assert np.isclose(classes[2].numpy(), 1.7079165), "Wrong value on classes"

    print("\033[92m All tests passed!")


def yolo_eval_test(target):
    tf.random.set_seed(10)
    yolo_outputs = (tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 2], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 1], mean=1, stddev=4, seed = 1),
                    tf.random.normal([19, 19, 5, 80], mean=1, stddev=4, seed = 1))
    scores, boxes, classes = target(yolo_outputs)

    assert type(scores) == EagerTensor, "Use tensoflow functions"
    assert type(boxes) == EagerTensor, "Use tensoflow functions"
    assert type(classes) == EagerTensor, "Use tensoflow functions"

    assert scores.shape == (10,), "Wrong shape"
    assert boxes.shape == (10, 4), "Wrong shape"
    assert classes.shape == (10,), "Wrong shape"
    
    assert np.isclose(scores[2].numpy(), 171.60194), "Wrong value on scores"
    assert np.allclose(boxes[2].numpy(), [-1240.3483, -3212.5881, -645.78, 2024.3052]), "Wrong value on boxes"
    assert np.isclose(classes[2].numpy(), 16), "Wrong value on classes"
    
    print("\033[92m All tests passed!")
