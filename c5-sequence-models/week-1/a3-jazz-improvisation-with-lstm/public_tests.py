from outputs import *
from dlai_tools.testing_utils import summary, comparator

def djmodel_test(model):
    comparator(summary(model), djmodel_output)

def music_inference_model_test(model):
    comparator(summary(model), music_inference_model_output)
