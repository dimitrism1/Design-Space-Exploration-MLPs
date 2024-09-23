import numpy as np
import qkeras.utils
import random

def generate_sparse(model,layer,l1,l2,sparsity):
    new = []
    with open("weight_pool.txt", 'r') as f:
        for lines in f:
            lines = float(lines)
            new.append(lines)
    new = np.array(new)
    bias = []
    with open("bias.txt", 'r') as f:
        for lines in f:
            lines = float(lines)
            bias.append(lines)
    bias = np.array(bias)
    
    zeros = np.zeros((int(l1*l2*sparsity),))
    x = np.array([])
    bias1 = np.array([])
    for i in range(int(l1*l2 - len(zeros))):
        random_index = random.randint(0,len(new) - 1)
        x = np.append(x,new[random_index])    
    y = np.concatenate((x,zeros))
    rng = np.random.default_rng()
    rng.shuffle(y)
    y = y.reshape(l1,l2)
    for i in range(l2):
        random_index_bias = random.randint(0,len(bias)-1)
        bias1 = np.append(bias1,bias[random_index_bias]) 
    model.weights[layer].assign(y)
    model.weights[layer + 1].assign(bias1)
    qkeras.utils.get_model_sparsity(model)
    return y