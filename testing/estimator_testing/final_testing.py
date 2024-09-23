import numpy as np
from testing_utils import generate_sparse
import tensorflow as tf
import hls4ml
import qkeras.utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, BatchNormalization
from tensorflow.keras.regularizers import l1
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import random
import os

def test_all(layer_num=50,bits = 8,int_bits=0,select='DSP',dirname = ""):
    rf_range = [1,2,3,4,10]
    models = []
    RF = []
    precision = []
    sparsity = []
    impl = {'LUT':0,'DSP':1}
    model_layers = [[] for x in range(layer_num)]
    initialiser = tf.keras.initializers.lecun_uniform()
    for prec in range(8,1,-1):
        x = 0
        for reuse in rf_range: 
            models = []
            RF = []
            precision = []
            sparsity = []
            selector = np.zeros(layer_num) + impl[select]
            print(selector)
            filename = "./" + dirname + "/results/prec_" + str(prec) + "/reuse_" + str(reuse) + "/model_"
            for i in range(x,layer_num,1):  
                    model = Sequential()
                    layer_1 = random.randint(8,25)
                    layer_2 = random.randint(4,60)                           
                    layer_3 = random.randint(4,64)                          
                    layer_4 = random.randint(8,40)
                    model_layers[i].append(layer_1)
                    model_layers[i].append(layer_2)
                    model_layers[i].append(layer_3)
                    model_layers[i].append(layer_4)

                    model = Sequential()
                    model.add(QDense(layer_2, input_shape=(layer_1,), name='fc1', kernel_quantizer=quantized_bits(bits,int_bits,alpha=1,use_stochastic_rounding=True),bias_quantizer=quantized_bits(bits,int_bits,alpha=1),
                                    kernel_initializer=initialiser, kernel_regularizer=l1(0.0001)   ))
                    model.add(QActivation(activation=quantized_relu(bits,int_bits,use_stochastic_rounding=False), name='relu1'))
                    model.add(QDense(layer_3, name='fc2',
                                    kernel_quantizer=quantized_bits(bits,int_bits,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(bits,int_bits,alpha=1),
                                    kernel_initializer=initialiser, kernel_regularizer=l1(0.0001 ) ))
                    model.add(QActivation(activation=quantized_relu(bits,int_bits,use_stochastic_rounding=False), name='relu2'))
                    model.add(QDense(layer_4, name='fc3',
                                     kernel_quantizer=quantized_bits(bits,int_bits,alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(bits,int_bits,alpha=1),
                                     kernel_initializer=initialiser, kernel_regularizer=l1(0.0001 ) ))

                    model.add(Activation(activation='softmax', name='softmax'))
                    sparsity.append(np.random.uniform(0.2,0.8))
                    for layer in range(0,len(model.layers),2):
                        generate_sparse(model,layer,model.weights[layer].shape[0],model.weights[layer].shape[1],sparsity[i])

                    models.append(model)
                    model.save(filename + str(i) + "/KERAS_check_best_mode.h5")
                    #RF.append(random.randint(1,3))
                    #RF.append(1)
                    RF.append(reuse)
                    precision.append(prec)

                    if not (os.path.exists(filename + str(i))):
                        os.makedirs(filename + str(i))
                    with open(filename + str(i) + "/layers.txt",'w') as f:
                        f.write(str(model_layers[i][0]) + " " + str(model_layers[i][1]))
                        f.close
                    with open(filename + str(i) + "/precision.txt",'w') as f:
                        f.write(str(precision[i]))
                        f.close

                    with open(filename + str(i) + "/reuse.txt",'w') as f:
                        f.write(str(int(RF[i])))
                        f.close




                    config = hls4ml.utils.config_from_keras_model(models[i],granularity='name')
                    #config['Model']['Precision']='fixed<16,4>'
                    config['LayerName']['fc1_input']['Precision']['result'] = 'fixed<' + str(precision[i]) + ',2>'
                    config['LayerName']['relu1']['Precision']['result'] = 'ap_ufixed<' + str(precision[i]) + ',0,AP_RND_CONV,AP_SAT>'
                    config['LayerName']['relu2']['Precision']['result'] = 'ap_ufixed<' + str(precision[i]) + ',0,AP_RND_CONV,AP_SAT>'
                    #config['LayerName']['relu3']['Precision']['result'] = 'ap_ufixed<' + str(precision[i]) + ',0,AP_RND_CONV,AP_SAT>'

                    #selector.append(random.randint(0,1))
                                #config['LayerName']['output']['Precision']['result'] = 'fixed<16,4>'
                    config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
                    config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
                    config['Model']['ReuseFactor'] = RF[i]
                    #config['LayerName']['fc1']['Precision']['result'] = 'fixed<16,4>'
                    #config['LayerName']['fc2']['Precision']['result'] = 'fixed<16,4>'
                    models[i].save(filename + str(i) + "/KERAS_check_best_mode.h5")
                    hls_model = hls4ml.converters.convert_from_keras_model(
                        models[i],hls_config = config,output_dir = filename + str(i) + '/hls4ml_prj',part = 'xc7z007s-clg225-2')
                    hls_model.compile()
                    os.system('cp -r ./scripts/project_orig.tcl ./scripts/build_prj_orig.tcl ./scripts/init.sh ./scripts/build_prj_orig_lut.tcl ./scripts/init_lut.sh ' + filename + str(i) + '/hls4ml_prj')
                    start_dir = os.getcwd()
                    os.chdir(filename + str(i) + '/hls4ml_prj')
                    if selector[i] == 0:
                        os.system('bash init_lut.sh')
                        os.system('rm init.sh')
                    elif selector[i] == 1:
                        os.system('bash init.sh')
                        os.system('rm init_lut.sh')
                    #os.chdir('/home/dmitsas/Downloads/notebooks/misc/')
                    os.chdir(start_dir)
                    if i==49:
                        for k in range(layer_num+1):
                            os.system("rm -r " + filename + str(k) + "/hls4ml_prj/myproject_prj/solution1/.autopilot")
                    with open(filename + str(i) + "/impl.txt",'w') as f:
                        f.write(str(selector[i]))
                        f.close

        for k in range(layer_num+1):
                os.system("rm -r " + filename + str(k) + "/hls4ml_prj/myproject_prj/solution1/.autopilot")

    return True