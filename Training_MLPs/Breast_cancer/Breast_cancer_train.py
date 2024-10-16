from tensorflow.keras.utils import to_categorical
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
import os
import numpy as np
import pandas as pd
seed=0
np.random.seed(seed)

from callbacks import all_callbacks

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1

from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import tensorflow.compat.v1 as tf1
from qkeras.utils import model_save_quantized_weights
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import random
from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from qkeras.utils import model_save_quantized_weights, load_qmodel
import hls4ml
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import qkeras.qtools.qtools_util
import qkeras.estimate

####Preprocess data to prepare it for training ####

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

df = pd.read_csv('./breast-cancer-wisconsin.csv', sep = ';')
X = df.drop('Y', axis = 1).values
y = df.Y
encoder = OneHotEncoder()
Y_reshaped = [[label] for label in y]
Y_encoded = encoder.fit_transform(Y_reshaped)
Y_encoded = Y_encoded.toarray()
X_train,X_test,Y_train,Y_test = train_test_split(X,Y_encoded,test_size=0.3,random_state=42)
sc = MinMaxScaler(feature_range=(0,0.9))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

id_num = 3
dirname = './models/models_' + str(id_num) 
#### if the layers are already generated the ready flag should be set to False in order to load them ######

ready = True
models = []
first_layer = []
sec_layer = []
model_num = 20
model_layers = [[] for x in range(model_num)]
bits = 8
int_bits = 0
for i in range(model_num):
    if ready:
        input1 = 10
        layer1 = random.randint(5,20)                           ###64
        first_layer.append(layer1)
        layer2 = random.randint(5,20)                           ###32
        sec_layer.append(layer2)
        output1 = 2
        model_layers[i].append(layer1)
        model_layers[i].append(layer2)
        if not (os.path.exists('layerdetails/models_' + str(id_num) + '/model_' + str(i))):
            os.makedirs('layerdetails/models_' + str(id_num) + '/model_' + str(i))
        with open('layerdetails/models_' + str(id_num) + '/model_' + str(i) + "/layers.txt",'w') as f:
            f.write(str(layer1) + " " + str(layer2))
        f.close
        #np.savetxt(dirname + str(i) + '/layerdetails.txt',model_layers[i],delimiter="\n" )
    else:
        with open('layerdetails/models_' + str(id_num) + '/model_' + str(i) + "/layers.txt",'r') as r:
            #print(r.readlines()[0])
            layer = r.readlines()[0].split(" ")
            input1 = 10
            layer1 = int(layer[0])
            layer2 = int(layer[1])
            output1 = 2
            first_layer.append(layer1)
            sec_layer.append(layer2)
    model=Sequential()
    model.add(QDense(layer1, input_shape=(input1,), name='fc1', kernel_quantizer=quantized_bits(8,0,alpha=1,use_stochastic_rounding=True),bias_quantizer=quantized_bits(bits,int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)   ))
    model.add(QActivation(activation=quantized_relu(bits,int_bits,use_stochastic_rounding=False), name='relu1'))
    model.add(QDense(layer2, name='fc2',
                    kernel_quantizer=quantized_bits(bits,int_bits,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(bits,int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
    model.add(QActivation(activation=quantized_relu(bits,int_bits,use_stochastic_rounding=False), name='relu2'))
    model.add(QDense(output1, name='output',
                    kernel_quantizer=quantized_bits(bits,int_bits,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(bits,int_bits,alpha=1),
                    kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
    model.add(Activation(activation='softmax', name='softmax'))
    models.append(model)
    
model_layers = np.array(model_layers)


######load pre trained models if the training flag is not activated
Train = True
if not Train:
    loaded_models = []
    pre = 8
    id_num = 2
    model_num = 20
    dirname = './models/models_' + str(id_num) 
    for model_no in range(model_num):
        co = {}
        _add_supported_quantized_objects(co)
        loaded_models.append(load_model(dirname + "/prec_" + str(pre) + "/model_" + str(model_no) + '/KERAS_check_best_model.h5', custom_objects=co))
        qkeras.utils.get_model_sparsity(loaded_models[model_no])
else:
    loaded_models = models

selector = 1       ###### 1 for DSP implementation and 0 for LUT implementation
for precision in range(8,1,-1):

    x = 0
    dirname = './models/models_' + str(id_num) + "/prec_" + str(precision) + "/model_"
    for i in range(x,model_num,1):    
        adam = Adam(lr=0.02)
        if(Train):
            loaded_models = models
            pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(0.2, begin_step=2000, frequency=100)}
            loaded_models[i] = prune.prune_low_magnitude(loaded_models[i], **pruning_params)
            loaded_models[i].compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
            callbacks= all_callbacks( outputDir = 'breast_cancer_new')
            callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())

            loaded_models[i].fit(X_train,Y_train,batch_size=1024,epochs=10,
                validation_split=0.25,
                verbose=1,
                shuffle=True,
                callbacks=callbacks.callbacks,
                )
            loaded_models[i] = strip_pruning(models[i])
            loaded_models[i].save(dirname + str(i) + '/KERAS_check_best_model.h5')

        rf_range = [1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100]
        for j in rf_range:
            loaded_models[i].save(dirname + str(i) + '/KERAS_check_best_model.h5')
            config = hls4ml.utils.config_from_keras_model(loaded_models[i],granularity='name')

            config['LayerName']['fc1_input']['Precision']['result'] = 'fixed<' + str(precision) + ',2>'
            config['LayerName']['relu1']['Precision']['result'] = 'ap_ufixed<' + str(precision) + ',0,AP_RND_CONV,AP_SAT>'
            config['LayerName']['relu2']['Precision']['result'] = 'ap_ufixed<' + str(precision) + ',0,AP_RND_CONV,AP_SAT>'
            
            config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
            config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
            config['Model']['ReuseFactor'] = j


            hls_model = hls4ml.converters.convert_from_keras_model(
                loaded_models[i],hls_config = config,output_dir = dirname + str(i) + '/reuse_' + str(j) + '/hls4ml_prj',part = 'xc7z007s-clg225-2')
            hls_model.compile()
            os.system('cp -r build_prj_orig.tcl project_orig.tcl build_prj_orig_lut.tcl ' + dirname + str(i) + '/reuse_' + str(j) + '/hls4ml_prj')
            os.system('cp init.sh init_lut.sh ' + dirname + str(i) + '/reuse_' + str(j) + '/hls4ml_prj')
            start_dir = os.getcwd()
            os.chdir(r'./models/models_'+ str(id_num) + "/prec_" + str(precision) + "/model_" + str(i) + '/reuse_' + str(j) + '/hls4ml_prj')
            if selector:
                os.system('bash init.sh')
                os.system('rm init_lut.sh')
            else:
                os.system('rm init.sh')
                os.system('bash init_lut.sh')
            os.chdir(start_dir)        
            if j==100:
                     for num in range(j+2):
                         os.system('rm -r ' + dirname +str(i) + '/reuse_' + str(num) + '/hls4ml_prj/myproject_prj/solution1/.autopilot')

