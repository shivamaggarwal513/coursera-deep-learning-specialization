from dlai_tools.testing_utils import summary, comparator


def happyModel_test(target):
    output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
              ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
              ['BatchNormalization', (None, 64, 64, 32), 128],
              ['ReLU', (None, 64, 64, 32), 0],
              ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
              ['Flatten', (None, 32768), 0],
              ['Dense', (None, 1), 32769, 'sigmoid']]
    
    comparator(summary(target()), output)

def convolutional_model_test(target):
    output = [['InputLayer', [(None, 64, 64, 3)], 0],
              ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
              ['ReLU', (None, 64, 64, 8), 0],
              ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
              ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
              ['ReLU', (None, 8, 8, 16), 0],
              ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
              ['Flatten', (None, 64), 0],
              ['Dense', (None, 6), 390, 'softmax']]
    
    comparator(summary(target((64, 64, 3))), output)
