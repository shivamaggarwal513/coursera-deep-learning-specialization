import copy
import numpy as np
from termcolor import colored


def clip_test(target, mValues):
    for mValue in mValues:
        print(f"\nGradients for mValue={mValue}")
        np.random.seed(3)
        dWax = np.random.randn(5, 3) * 10
        dWaa = np.random.randn(5, 5) * 10
        dWya = np.random.randn(2, 5) * 10
        dba = np.random.randn(5, 1) * 10
        dby = np.random.randn(2, 1) * 10
        gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "dba": dba, "dby": dby}

        gradients2 = target(gradients, mValue)
        print("gradients[\"dWaa\"][1][2] =", gradients2["dWaa"][1][2])
        print("gradients[\"dWax\"][3][1] =", gradients2["dWax"][3][1])
        print("gradients[\"dWya\"][1][2] =", gradients2["dWya"][1][2])
        print("gradients[\"dba\"][4] =", gradients2["dba"][4])
        print("gradients[\"dby\"][1] =", gradients2["dby"][1])
        
        for grad in gradients2.keys():
            valuei = gradients[grad]
            valuef = gradients2[grad]
            mink = np.min(valuef)
            maxk = np.max(valuef)
            assert mink >= -abs(mValue), f"Problem with {grad}. Set a_min to -mValue in the np.clip call"
            assert maxk <= abs(mValue), f"Problem with {grad}.Set a_max to mValue in the np.clip call"
            index_not_clipped = np.logical_and(valuei <= mValue, valuei >= -mValue)
            assert np.all(valuei[index_not_clipped] == valuef[index_not_clipped]), f" Problem with {grad}. Some values that should not have changed, changed during the clipping process."
        
        print(colored("All tests passed!", "light_green"))

def sample_test(target, char_to_ix, ix_to_char):
    np.random.seed(24)
    vocab_size, n_a = 27, 100
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    ba, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}

    indices = target(parameters, char_to_ix, 0)
    print("Sampling:")
    print("list of sampled indices:\n", indices)
    print("list of sampled characters:\n", [ix_to_char[i] for i in indices])
    
    assert len(indices) < 52, "Indices length must be smaller than 52"
    assert indices[-1] == char_to_ix['\n'], "All samples must end with \\n"
    assert min(indices) >= 0 and max(indices) < len(char_to_ix), f"Sampled indexes must be between 0 and len(char_to_ix)={len(char_to_ix)}"
    assert np.allclose(indices[0:6], [23, 16, 26, 26, 24, 3]), "Wrong values"
    
    print(colored("All tests passed!", "light_green"))

def optimize_test(target):
    np.random.seed(1)
    vocab_size, n_a = 27, 100
    a_prev = np.random.randn(n_a, 1)
    Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
    ba, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "ba": ba, "by": by}
    X = [12, 3, 5, 11, 22, 3]
    Y = [4, 14, 11, 22, 25, 26]
    old_parameters = copy.deepcopy(parameters)
    loss, gradients, a_last = target(X, Y, a_prev, parameters, learning_rate = 0.01)
    print("Loss =", loss)
    print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
    print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
    print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
    print("gradients[\"dba\"][4] =", gradients["dba"][4])
    print("gradients[\"dby\"][1] =", gradients["dby"][1])
    print("a_last[4] =", a_last[4])
    
    assert np.isclose(loss, 126.5039757), "Problems with the call of the rnn_forward function"
    for grad in gradients.values():
        assert np.min(grad) >= -5, "Problems in the clip function call"
        assert np.max(grad) <= 5, "Problems in the clip function call"
    assert np.allclose(gradients['dWaa'][1, 2], 0.1947093), "Unexpected gradients. Check the rnn_backward call"
    assert np.allclose(gradients['dWya'][1, 2], -0.007773876), "Unexpected gradients. Check the rnn_backward call"
    assert not np.allclose(parameters['Wya'], old_parameters['Wya']), "parameters were not updated"
    
    print(colored("All tests passed!", "light_green"))

def model_test(last_name):
    assert last_name == 'Trodonosaurus\n', "Wrong expected output"
    print(colored("All tests passed!", "light_green"))
