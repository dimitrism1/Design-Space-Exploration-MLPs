from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import qkeras.qtools.qtools_util
import qkeras.estimate
import qkeras.utils
import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt

class regression():
    #def __init__(self):
    model_num = 20
    
    @classmethod  
    def load(self,neural_name,dirname = "",num = model_num):
        self.id_num = {'jet_tagging':1,'Arrythmia':1,'Breast_cancer':0,"Cardio":0}
        NN_name =neural_name # "Breast_cancer"
        if NN_name[0].islower() and NN_name != 'jet_tagging':
            NN_name = NN_name[0].upper() + NN_name[1:]
        if dirname == "":
        	self.dirname = './' + NN_name + "/models/models_" + str(self.id_num[NN_name]) + "/prec_" + str(8) + '/model_'

        else:
        	self.dirname = dirname
        self.model_num = num
        loaded_models = []

        for model_no in range(self.model_num):
            co = {}
            _add_supported_quantized_objects(co)
            loaded_models.append(load_model(self.dirname + str(model_no) + '/KERAS_check_best_model.h5', custom_objects=co))


            qkeras.utils.get_model_sparsity(loaded_models[model_no])

        return loaded_models
    
    
    @classmethod
    def print_resource(self,neural_name,precision=8,resource = 'FF',model = "",dirname = ""):
        
        self.model_num = 20
        self.max_RF = 100
        self.id_num = {'jet_tagging':1,'Arrythmia':1,'Breast_cancer':0,"Cardio":0}
        NN_name =neural_name # "Breast_cancer"
        if NN_name[0].islower() and NN_name != 'jet_tagging':
            NN_name = NN_name[0].upper() + NN_name[1:]
        #if NN_name in self.id_num:
        if dirname == "":
        	self.dirname = './' + NN_name + "/models/models_" + str(self.id_num[NN_name]) + "/prec_" + str(precision) + '/model_'
        else:
        	self.dirname = dirname
        if model == "":
            loaded_models = self.load(NN_name)
        else:
            loaded_models = model  ###### TODO: error handling in case the argument is not a valid model
        #else:
            #self.dirname = dir_name
        selector = {'DSP':11,'FF':12,'LUT':13}
        res = [[[] for x in range(self.model_num)] for y in range(int(len(loaded_models[0].layers)/2))]
        for layer_num in range(1):
            for it in range(self.model_num):
                for RF in range(10):
                    with open(self.dirname + str(it) + '/reuse_' + str(RF+1) + '/hls4ml_prj/myproject_prj/solution1/syn/report/csynth.rpt') as f:
                        for i,lines in enumerate(f):
                            if(i == 22):
                                #print(lines) 
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num][it].append(0)
                                else:
                                    res[layer_num][it].append(0)
                                #FF.append(d[selector[resource]].split(" ")[2])
                                #break
                            if(i==24):
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num + 1][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num + 1][it].append(0)
                                else:
                                    res[layer_num + 1][it].append(0)
                                if len(loaded_models[0].layers) == 4:
                                    break
                            if(i == 26):
                                #print(lines) 
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num + 2][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num + 2][it].append(0)
                                else:
                                    res[layer_num + 2][it].append(0)
                                if len(loaded_models[0].layers) == 6:
                                    break
                                #FF.append(d[selector[resource]].split(" ")[2])
                                #break
                            if(i==28):
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num + 3][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num + 3][it].append(0)
                                else:
                                    res[layer_num + 3][it].append(0)
                                break
        for layer_num in range(1):
            for it in range(self.model_num):
                for RF in range(19,self.max_RF,10):
                    with open(self.dirname + str(it) + '/reuse_' + str(RF+1) + '/hls4ml_prj/myproject_prj/solution1/syn/report/csynth.rpt') as f:
                        for i,lines in enumerate(f):
                            #if(counter == 24):
                            if(i == 22):
                                #print(lines) 
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num][it].append(0)
                                else:
                                    res[layer_num][it].append(0)
                                #FF.append(d[selector[resource]].split(" ")[2])
                                #break
                                
                            if(i==24):
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num + 1][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num + 1][it].append(0)
                                else:
                                    res[layer_num + 1][it].append(0)
                                if len(loaded_models[0].layers) == 4:
                                    break
                            if(i == 26):
                                #print(lines) 
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num + 2][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num + 2][it].append(0)
                                else:
                                    res[layer_num + 2][it].append(0)
                                #FF.append(d[selector[resource]].split(" ")[2])
                                #break
                                if len(loaded_models[0].layers) == 6:
                                    break
                            if(i==28):
                                d = lines.split("|")
                                #if((d[selector[resource]].replace(" ",""))!='-'):
                                if len(d) > 1:
                                    if(d[selector[resource]].replace(" ","")!='-'):
                                        res[layer_num + 3][it].append(int(d[selector[resource]].replace(" ","").split("(")[0]))
                                    else:
                                        res[layer_num + 3][it].append(0)
                                else:
                                    res[layer_num + 3][it].append(0)
                                break




        res = np.array(res)
        return res
    @classmethod
    def get_muls(self,models,model_num = 20):
        loaded_models = models
        neurons = [[] for x in range(model_num)]
        inputs = [[] for x in range(model_num)]
        for num in range(model_num):
            for layer_num in range(0,len(loaded_models[num].layers),2):
                neurons[num].append(loaded_models[num].layers[layer_num].output.shape[1])
                inputs[num].append(loaded_models[num].layers[layer_num].input.shape[1])
        
        muls = [[] for x in range(int(len(loaded_models[0].layers) /2))]
        zeros = [[] for x in range(int(len(loaded_models[0].layers) /2))]
        for i in range(model_num):
            for layer_num in range(int(len(loaded_models[0].layers) /2)):
                muls[layer_num].append(inputs[i][layer_num]*neurons[i][layer_num])
                zero = 0
                for j in loaded_models[i].weights[layer_num*2]:
                    zero += len(j) - np.count_nonzero(j)

                zeros[layer_num].append(zero)
                
        real_muls = [[] for x in range(int(len(loaded_models[0].layers) /2))]

        for i in range(model_num):
            for layer_num in range(int(len(loaded_models[0].layers) /2)):
                real_muls[layer_num].append(muls[layer_num][i] - zeros[layer_num][i])
        return real_muls
    @classmethod	
    def get_all_muls(self,models,model_num = 20):
    	loaded_models = models
    	neurons = [[] for x in range(model_num)]
    	inputs = [[] for x in range(model_num)]
    	for num in range(model_num):
            	for layer_num in range(0,len(loaded_models[num].layers),2):
                	neurons[num].append(loaded_models[num].layers[layer_num].output.shape[1])
                	inputs[num].append(loaded_models[num].layers[layer_num].input.shape[1])
	
	
    	muls = [[] for x in range(int(len(loaded_models[0].layers) /2))]
    	zeros = [[] for x in range(int(len(loaded_models[0].layers) /2))]
    	for i in range(model_num):
            	for layer_num in range(int(len(loaded_models[0].layers) /2)):	
                	muls[layer_num].append(inputs[i][layer_num]*neurons[i][layer_num])
    	return muls
         
       
             
