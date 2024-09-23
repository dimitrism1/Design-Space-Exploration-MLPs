from typing import List,Callable
from collections import namedtuple
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
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
import pymoo

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from callbacks import all_callbacks



import create_pop as crp
import estimator.estimator as est


class dse(Problem):
    def __init__(self,layer_in,layer_out,X_train,X_test,Y_train,Y_test,test_compare,population ="",parameters = "",dirname = "./dse_model"):
        self.layer_in = layer_in
        self.layer_out = layer_out
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.population = population
        self.parameters = parameters
        self.test_compare = test_compare
        self.dirname = dirname
        
        if type(self.parameters) == np.ndarray:
            self.ready = True
        else:
            self.ready = False
        super().__init__(n_var = 8,
                         n_obj = 2,
                         xl = np.array([10,8,6,2,0,10,2,0]),
                         xu = np.array([64,64,32,8,1,95,8,1]),
                         vtype = "int"
                        )
    def _evaluate(self,x,out):
        if type(self.population) == np.ndarray:
            
            accuracy = self.population[0,:]
            F1 = np.ones(len(accuracy)) - accuracy
            F2 = self.population[1,:]


        else:
            accuracy,model = (crp.create_pop(self.layer_in,x[:,0],x[:,1],x[:,2],self.layer_out,x[:,3],x[:,4],x[:,5],x[:,6],x[:,7],self.X_train,self.X_test,self.Y_train,self.Y_test,self.test_compare,dirname=self.dirname))
            F1 = np.ones(len(accuracy)) - accuracy
            print(x)
            F2 = est.estimator.estim_model(model = model,precision = x[:,6],int_bits = x[:,4],reuse=1,DSP_mul=x[:,7],suppress=True,dirname = self.dirname)
            
        print("F1 is " + str(F1))        
        print("F2 is " + str(F2))
        
        out['F'] = [F1,F2]
        
        
    def minimize(self,problem,pop_size,generations,seed=1,Verbose = True):
        if not self.ready:
            algorithm = NSGA2(pop_size = pop_size)
        else:
            algorithm = NSGA2(pop_size = pop_size,sampling = self.parameters,eliminate_duplicates=False)
            print(self.parameters)
        self.res = minimize(problem,algorithm,generations,seed,Verbose=True)
        return self.res
    
    def get_opt_solutions(self):
        return self.res.F
    
    def get_opt_params(self):
        return self.res.X.round().astype(int)
    
    def get_all_solutions(self):
        pop = self.res.pop
        solutions = pop.get("F")
        return solutions
    
    def get_all_params(self):
        pop = self.res.pop
        params  = pop.get("X").round().astype(int) if self.ready != 1 else self.parameters
        return params
    
    def get_fitting_solutions(self):
        return self.res.F[self.res.F[:,1] < 1]
    def get_fitting_parameters(self):
        return self.res.X[self.res.F[:,1] < 1].round().astype(int)
        
    def plot_parento_front(self):
        plt.scatter(self.get_all_solutions()[:,0],self.get_all_solutions()[:,1])
        plt.scatter(self.get_opt_solutions()[:,0],self.get_opt_solutions()[:,1])
        plt.ylabel("Resources")
        plt.xlabel("1 - Accuracy")
        
