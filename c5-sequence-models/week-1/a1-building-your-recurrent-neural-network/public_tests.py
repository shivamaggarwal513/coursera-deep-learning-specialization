import numpy as np
from termcolor import colored
from rnn_utils import *
from test_cases import *


def rnn_cell_forward_test(target):
    m, n_x, n_a, n_y = 10, 3, 5, 2
    
    # Only bias in expression
    xt, a_prev, parameters = rnn_cell_forward_test_case(m, n_x, n_a, n_y)
    xt = np.zeros((n_x, m))
    a_prev = np.zeros((n_a, m))
    
    # setting wya to zero to catch if a learner has used "a_prev" for "yt_pred"
    parameters['Wya'] = np.zeros((n_y, n_a))

    a_next, yt_pred, cache = target(xt, a_prev, parameters)
    
    assert a_next.shape == (n_a, m), f"Wrong shape for a_next. Expected ({n_a, m}) != {a_next.shape}"
    assert yt_pred.shape == (n_y, m), f"Wrong shape for yt_pred. Expected ({n_y, m}) != {yt_pred.shape}"
    assert cache[0].shape == (n_a, m), "Wrong shape in cache->a_next"
    assert cache[1].shape == (n_a, m), "Wrong shape in cache->a_prev"
    assert cache[2].shape == (n_x, m), "Wrong shape in cache->x_t"
    assert len(cache[3].keys()) == 5, "Wrong number of parameters in cache. Expected 5"
    
    assert np.allclose(np.tanh(parameters['ba']), a_next), "Problem 1 in a_next expression. Related to ba?"
    assert np.allclose(softmax(parameters['by']), yt_pred), "Problem 1 in yt_pred expression. Related to by?"

    # Only xt in expression
    a_prev = np.zeros((n_a, m))
    xt = np.random.randn(n_x, m)
    parameters['Wax'] = np.random.randn(n_a, n_x)
    # reinitialize wya with random values for the remaining tests
    parameters['Wya'] = np.random.randn(n_y, n_a)
    parameters['ba'] = np.zeros((n_a, 1))
    parameters['by'] = np.zeros((n_y, 1))

    a_next, yt_pred, cache = target(xt, a_prev, parameters)

    assert np.allclose(np.tanh(np.dot(parameters['Wax'], xt)), a_next), "Problem 2 in a_next expression. Related to xt?"
    assert np.allclose(softmax(np.dot(parameters['Wya'], a_next)), yt_pred), "Problem 2 in yt_pred expression. Related to a_next?"

    # Only a_prev in expression
    a_prev = np.random.randn(n_a, m)
    xt = np.zeros((n_x, m))
    parameters['Waa'] = np.random.randn(n_a, n_a)
    parameters['ba'] = np.zeros((n_a, 1))
    parameters['by'] = np.zeros((n_y, 1))

    a_next, yt_pred, cache = target(xt, a_prev, parameters)

    assert np.allclose(np.tanh(np.dot(parameters['Waa'], a_prev)), a_next), "Problem 3 in a_next expression. Related to a_prev?"
    assert np.allclose(softmax(np.dot(parameters['Wya'], a_next)), yt_pred), "Problem 3 in yt_pred expression. Related to a_next?"

    print(colored("All tests passed", "light_green"))


def rnn_forward_test(target):
    T_x, m, n_x, n_a, n_y = 13, 8, 4, 7, 3
    x, a0, parameters = rnn_forward_test_case(T_x, m, n_x, n_a, n_y, seed=17)
    
    a, y_pred, caches = target(x, a0, parameters)
    
    assert a.shape == (n_a, m, T_x), f"Wrong shape for a. Expected: ({n_a, m, T_x}) != {a.shape}"
    assert y_pred.shape == (n_y, m, T_x), f"Wrong shape for y_pred. Expected: ({n_y, m, T_x}) != {y_pred.shape}"
    assert len(caches[0]) == T_x, f"len(cache) must be T_x = {T_x}"
    
    assert np.allclose(a[5, 2, 2:6], [0.99999291, 0.99332189, 0.9921928, 0.99503445]), "Wrong values for a"
    assert np.allclose(y_pred[2, 1, 1: 5], [0.19428, 0.14292, 0.24993, 0.00119], atol=1e-4), "Wrong values for y_pred"
    assert np.allclose(caches[1], x), f"Fail check: cache[1] != x"
    
    print(colored("All tests passed", "light_green"))


def lstm_cell_forward_test(target):
    m, n_x, n_a, n_y = 8, 4, 7, 3
    x, a0, c0, parameters = lstm_cell_forward_test_case(m, n_x, n_a, n_y, seed=212)
    
    a_next, c_next, y_pred, cache = target(x, a0, c0, parameters)
    
    assert len(cache) == 10, "Don't change the cache"
    
    assert cache[4].shape == (n_a, m), f"Wrong shape for cache[4](ft). {cache[4].shape} != {(n_a, m)}"
    assert cache[5].shape == (n_a, m), f"Wrong shape for cache[5](it). {cache[5].shape} != {(n_a, m)}"
    assert cache[6].shape == (n_a, m), f"Wrong shape for cache[6](cct). {cache[6].shape} != {(n_a, m)}"
    assert cache[1].shape == (n_a, m), f"Wrong shape for cache[1](c_next). {cache[1].shape} != {(n_a, m)}"
    assert cache[7].shape == (n_a, m), f"Wrong shape for cache[7](ot). {cache[7].shape} != {(n_a, m)}"
    assert cache[0].shape == (n_a, m), f"Wrong shape for cache[0](a_next). {cache[0].shape} != {(n_a, m)}"
    assert cache[8].shape == (n_x, m), f"Wrong shape for cache[8](xt). {cache[8].shape} != {(n_x, m)}"
    assert cache[2].shape == (n_a, m), f"Wrong shape for cache[2](a_prev). {cache[2].shape} != {(n_a, m)}"
    assert cache[3].shape == (n_a, m), f"Wrong shape for cache[3](c_prev). {cache[3].shape} != {(n_a, m)}"
    
    assert a_next.shape == (n_a, m), f"Wrong shape for a_next. {a_next.shape} != {(n_a, m)}"
    assert c_next.shape == (n_a, m), f"Wrong shape for c_next. {c_next.shape} != {(n_a, m)}"
    assert y_pred.shape == (n_y, m), f"Wrong shape for y_pred. {y_pred.shape} != {(n_y, m)}"
    
    assert np.allclose(cache[4][0, 0:2], [0.32969833, 0.0574555]), "wrong values for ft"
    assert np.allclose(cache[5][0, 0:2], [0.0036446, 0.9806943]), "wrong values for it"
    assert np.allclose(cache[6][0, 0:2], [0.99903873, 0.57509956]), "wrong values for cct"
    assert np.allclose(cache[1][0, 0:2], [0.1352798,  0.39884899]), "wrong values for c_next"
    assert np.allclose(cache[7][0, 0:2], [0.7477249,  0.71588751]), "wrong values for ot"
    assert np.allclose(cache[0][0, 0:2], [0.10053951, 0.27129536]), "wrong values for a_next"
    
    assert np.allclose(y_pred[1], [0.417098, 0.449528, 0.223159, 0.278376,
                                   0.68453,  0.419221, 0.564025, 0.538475]), "Wrong values for y_pred"
    
    print(colored("All tests passed", "light_green"))


def lstm_forward_test(target):
    T_x, m, n_x, n_a, n_y = 16, 13, 4, 3, 2
    x, a0, parameters = lstm_forward_test_case(T_x, m, n_x, n_a, n_y, seed=45)

    a, y, c, caches = target(x, a0, parameters)
    
    assert a.shape == (n_a, m, T_x), f"Wrong shape for a. {a.shape} != {(n_a, m, T_x)}"
    assert c.shape == (n_a, m, T_x), f"Wrong shape for c. {c.shape} != {(n_a, m, T_x)}"
    assert y.shape == (n_y, m, T_x), f"Wrong shape for y. {y.shape} != {(n_y, m, T_x)}"
    assert len(caches[0]) == T_x, f"Wrong shape for caches. {len(caches[0])} != {T_x} "
    assert len(caches[0][0]) == 10, f"length of caches[0][0] must be 10."
    
    assert np.allclose(a[2, 1, 4:6], [-0.01606022,  0.0243569]), "Wrong values for a"
    assert np.allclose(c[2, 1, 4:6], [-0.02753855,  0.05668358]), "Wrong values for c"
    assert np.allclose(y[1, 1, 4:6], [0.70444592 ,0.70648935]), "Wrong values for y"
    
    print(colored("All tests passed", "light_green"))
