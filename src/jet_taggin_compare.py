from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
seed = 0
np.random.seed(seed)
import tensorflow as tf
tf.random.set_seed(seed)
import os


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from callbacks import all_callbacks

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import tensorflow.compat.v1 as tf1
from qkeras.utils import model_save_quantized_weights

from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning


from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
import math
from sklearn.datasets import fetch_openml


from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras.utils import model_save_quantized_weights, load_qmodel
import hls4ml
import qkeras.utils


from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import pymoo

def compare_data(hls_model,X_test,Y_test):
    X_test = np.ascontiguousarray(X_test)
    y_p = hls_model.predict(X_test)
    y_pred = to_categorical(np.argmax(y_p,axis=1),5)
    acc = accuracy_score(y_pred,Y_test)
    return acc