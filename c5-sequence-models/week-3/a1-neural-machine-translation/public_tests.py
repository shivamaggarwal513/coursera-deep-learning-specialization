import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
from termcolor import colored
from dlai_tools.testing_utils import summary, comparator


def one_step_attention_test(target):
    np.random.seed(10)
    tf.random.set_seed(10)

    m, Tx, n_a, n_s = 10, 30, 32, 64
    
    a = np.random.uniform(1, 0, (m, Tx, 2 * n_a)).astype(np.float32)
    s_prev = np.random.uniform(1, 0, (m, n_s)).astype(np.float32)
    context = target(a, s_prev)
    
    expected_output = np.load('data/one_step_attention_output.npy')
    
    assert type(context) == EagerTensor, "Unexpected type. It should be a Tensor"
    assert tuple(context.shape) == (m, 1, n_s), "Unexpected output shape"
    assert np.all(context.numpy() == expected_output), "Unexpected values in the result"
    
    print(colored("All tests passed!", "light_green"))


def modelf_test(target):
    Tx, Ty, n_a, n_s = 30, 10, 32, 64
    len_human_vocab = 37
    len_machine_vocab = 11
    
    model = target(Tx, Ty, n_a, n_s, len_human_vocab, len_machine_vocab)
    
    expected_summary = [['InputLayer', [(None, 30, 37)], 0],
                        ['InputLayer', [(None, 64)], 0],
                        ['Bidirectional', (None, 30, 64), 17920],
                        ['RepeatVector', (None, 30, 64), 0, 30],
                        ['Concatenate', (None, 30, 128), 0],
                        ['Dense', (None, 30, 10), 1290, 'tanh'],
                        ['Dense', (None, 30, 1), 11, 'relu'],
                        ['Activation', (None, 30, 1), 0],
                        ['Dot', (None, 1, 64), 0],
                        ['InputLayer', [(None, 64)], 0],
                        ['LSTM',[(None, 64), (None, 64), (None, 64)], 33024,[(None, 1, 64), (None, 64), (None, 64)],'tanh'],
                        ['Dense', (None, 11), 715, 'softmax']]
    
    assert len(model.outputs) == 10, f"Wrong output shape. Expected 10 != {len(model.outputs)}"
    
    comparator(summary(model), expected_summary)

def model_compile_test(model, opt):
    assert opt.lr == 0.005, "Set the lr parameter to 0.005"
    assert opt.beta_1 == 0.9, "Set the beta_1 parameter to 0.9"
    assert opt.beta_2 == 0.999, "Set the beta_2 parameter to 0.999"
    assert model.loss == "categorical_crossentropy", "Wrong loss. Use 'categorical_crossentropy'"
    assert model.optimizer == opt, "Use the optimizer that you have instantiated"
    assert model.compiled_metrics._user_metrics[0] == 'accuracy', "set metrics to ['accuracy']"

    print(colored("All tests passed!", "light_green"))
