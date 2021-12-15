import torch
import tensorflow as tf
import numpy as np
from submissions.load_submission import load_submission
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # FATAL


def count_parameters_torch(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_tf(model):
    return np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])


print(count_parameters_torch(load_submission("goudarzi")))
print(count_parameters_torch(load_submission("rothlubbers")))
