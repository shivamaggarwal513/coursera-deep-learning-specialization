import numpy as np


def rnn_forward_parameters(n_x, n_a, n_y):
    return {
        'Waa': np.random.randn(n_a, n_a),
        'Wax': np.random.randn(n_a, n_x),
        'Wya': np.random.randn(n_y, n_a),
        'ba': np.random.randn(n_a, 1),
        'by': np.random.randn(n_y, 1)
    }


def rnn_cell_forward_test_case(m, n_x, n_a, n_y, seed = 1):
    np.random.seed(seed)
    xt = np.random.randn(n_x, m)
    a_prev = np.random.randn(n_a, m)
    parameters = rnn_forward_parameters(n_x, n_a, n_y)
    return xt, a_prev, parameters


def rnn_forward_test_case(T_x, m, n_x, n_a, n_y, seed = 1):
    np.random.seed(seed)
    x = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)
    parameters = rnn_forward_parameters(n_x, n_a, n_y)
    return x, a0, parameters


def lstm_forward_parameters(n_x, n_a, n_y):
    return {
        'Wf': np.random.randn(n_a, n_a + n_x),
        'bf': np.random.randn(n_a, 1),
        'Wi': np.random.randn(n_a, n_a + n_x),
        'bi': np.random.randn(n_a, 1),
        'Wo': np.random.randn(n_a, n_a + n_x),
        'bo': np.random.randn(n_a, 1),
        'Wc': np.random.randn(n_a, n_a + n_x),
        'bc': np.random.randn(n_a, 1),
        'Wy': np.random.randn(n_y, n_a),
        'by': np.random.randn(n_y, 1)
    }


def lstm_cell_forward_test_case(m, n_x, n_a, n_y, seed = 1):
    np.random.seed(seed)
    xt = np.random.randn(n_x, m)
    a_prev = np.random.randn(n_a, m)
    c_prev = np.random.randn(n_a, m)
    parameters = lstm_forward_parameters(n_x, n_a, n_y)
    return xt, a_prev, c_prev, parameters


def lstm_forward_test_case(T_x, m, n_x, n_a, n_y, seed = 1):
    np.random.seed(seed)
    x = np.random.randn(n_x, m, T_x)
    a0 = np.random.randn(n_a, m)
    parameters = lstm_forward_parameters(n_x, n_a, n_y)
    return x, a0, parameters
