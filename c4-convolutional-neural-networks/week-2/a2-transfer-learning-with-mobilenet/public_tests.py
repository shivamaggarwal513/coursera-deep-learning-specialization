import tensorflow as tf
from dlai_tools.testing_utils import summary, comparator
from termcolor import colored

IMG_SIZE = (160, 160)
IMG_SHAPE = IMG_SIZE + (3,)


def data_augmenter_test(target):
    augmenter = target()
    assert augmenter.layers[0].name.startswith('random_flip'), "First layer must be RandomFlip"
    assert augmenter.layers[0].mode == 'horizontal', "RadomFlip parameter must be horizontal"
    assert augmenter.layers[1].name.startswith('random_rotation'), "Second layer must be RandomRotation"
    assert augmenter.layers[1].factor == 0.2, "Rotation factor must be 0.2"
    assert len(augmenter.layers) == 2, "The model must have only 2 layers"

    print(colored("All tests passed!", "green"))


def alpaca_model_test(model):
    alpaca_summary = [['InputLayer', [(None, 160, 160, 3)], 0],
                      ['Sequential', (None, 160, 160, 3), 0],
                      ['TFOpLambda', (None, 160, 160, 3), 0],
                      ['TFOpLambda', (None, 160, 160, 3), 0],
                      ['Functional', (None, 5, 5, 1280), 2257984],
                      ['GlobalAveragePooling2D', (None, 1280), 0],
                      ['Dropout', (None, 1280), 0, 0.2],
                      ['Dense', (None, 1), 1281, 'linear']]
    comparator(summary(model), alpaca_summary)


def transfer_learning_test(loss_function, optimizer, metrics):
    assert type(loss_function) == tf.keras.losses.BinaryCrossentropy, "Not the correct layer"
    assert loss_function.from_logits, "Use from_logits=True"
    assert type(optimizer) == tf.keras.optimizers.Adam, "This is not an Adam optimizer"
    assert optimizer.lr == 0.0001, "Wrong learning rate"
    assert metrics[0] == 'accuracy', "Wrong metric"

    print(colored("All tests passed!", "green"))
