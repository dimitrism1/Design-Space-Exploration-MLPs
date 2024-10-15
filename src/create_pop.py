def create_pop(layer_1,layer_2,layer_3,layer_4,layer_5,bits,int_bits,sparsity,precision,selector,X_train,X_test,Y_train,Y_test,compare_test,batch_size,dirname = "./dse_models/models_115" + "/model_"):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1
    from callbacks import all_callbacks
    from tensorflow.keras.layers import Activation
    from qkeras.qlayers import QDense, QActivation
    from qkeras.quantizers import quantized_bits, quantized_relu
    import tensorflow as tf
    import random
    import math
    import os
    from tensorflow.python.client import device_lib
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import r2_score
    
    from tensorflow_model_optimization.python.core.sparsity.keras import prune, pruning_callbacks, pruning_schedule
    from tensorflow_model_optimization.sparsity.keras import strip_pruning
    from qkeras.utils import model_save_quantized_weights, load_qmodel
    import hls4ml
    import qkeras.utils
    from callbacks import all_callbacks
    from tensorflow.keras.utils import to_categorical
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    import numpy as np
    
    
    
    #id_num = 2
    #dirname = "./dse_models/models_" + str(id_num) + "/model_"
    models = []
    accuracy = []
    RF = 1
    if type(bits) == np.ndarray:
        bits = bits.round().astype(int)
        int_bits = int_bits.round().astype(int)
        precision = precision.round().astype(int)
        selector = selector.round()
        layer_2 = layer_2.round().astype(int)
        layer_3 = layer_3.round().astype(int)
        layer_4 = layer_4.round().astype(int)
        case_select = 1
        bits_length = len(bits)
    else:
        bits = np.array([bits]).astype(float)
        int_bits = np.array([int_bits]).astype(int)
        precision = np.array([precision]).astype(int)
        sparsity = np.array([sparsity]).astype(int)
        selector = selector.round()
        layer_2 = np.array([layer_2]).astype(int)
        layer_3 = np.array([layer_3]).astype(int)
        layer_4 = np.array([layer_4]).astype(int)
        case_select = 0
        bits_length = 1
    for i in range(bits_length):
        model = Sequential()

        model.add(QDense(layer_2[i], input_shape=(layer_1,), name='fc1', kernel_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1,use_stochastic_rounding=True),bias_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1),
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001)   ))
        model.add(QActivation(activation=quantized_relu(int(bits[i]),int_bits[i],use_stochastic_rounding=False), name='relu1'))
        model.add(QDense(layer_3[i], name='fc2',
                        kernel_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1),
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
        model.add(QActivation(activation=quantized_relu(int(bits[i]),int_bits[i],use_stochastic_rounding=False), name='relu2'))
        model.add(QDense(layer_4[i], name='fc3',
                        kernel_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1),
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
        model.add(QActivation(activation=quantized_relu(int(bits[i]),int_bits[i],use_stochastic_rounding=False), name='relu3'))
        model.add(QDense(layer_5, name='output',
                        kernel_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1,use_stochastic_rounding=True), bias_quantizer=quantized_bits(int(bits[i]),int_bits[i],alpha=1),
                        kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001 ) ))
        model.add(Activation(activation='softmax', name='softmax'))
        adam = Adam(lr=0.02)
        Train = True
        if(Train):
            pruning_params = {"pruning_schedule" : pruning_schedule.ConstantSparsity(sparsity[i]/100, begin_step=2000, frequency=100)}
            model = prune.prune_low_magnitude(model, **pruning_params)

            model.compile(optimizer=adam, loss=['categorical_crossentropy'], metrics=['accuracy'])
            callbacks = all_callbacks( outputDir = 'arrythmia_classification_prune_new')
            callbacks.callbacks.append(pruning_callbacks.UpdatePruningStep())


            model.fit(X_train,Y_train,batch_size=batch_size,epochs=10,
                validation_split=0.25,
                verbose=1,
                shuffle=True,
                callbacks=callbacks.callbacks,
                )
            model = strip_pruning(model)
            #model_save_quantized_weights(model, "test_weights_new_" + str(i) + "_" + str(id_num))
            model.save(dirname + str(i) + '/KERAS_check_best_model.h5')
            saved_model = model
        models.append(model)
            
        config = hls4ml.utils.config_from_keras_model(model,granularity='name')
        config['LayerName']['fc1_input']['Precision']['result'] = 'fixed<' + str(precision[i]) + ',2>'
        config['LayerName']['relu1']['Precision']['result'] = 'fixed<' + str(precision[i]) + ',0,AP_RND_CONV,AP_SAT>'
        config['LayerName']['relu2']['Precision']['result'] = 'fixed<' + str(precision[i]) + ',0,AP_RND_CONV,AP_SAT>'
        config['LayerName']['relu3']['Precision']['result'] = 'fixed<' + str(precision[i]) + ',0,AP_RND_CONV,AP_SAT>'

        config['LayerName']['softmax']['exp_table_t'] = 'ap_fixed<18,8>'
        config['LayerName']['softmax']['inv_table_t'] = 'ap_fixed<18,4>'
        config['Model']['ReuseFactor'] = RF
            #config['LayerName']['fc1']['Precision']['result'] = 'fixed<16,4>'
            #config['LayerName']['fc2']['Precision']['result'] = 'fixed<16,4>'

        hls_model = hls4ml.converters.convert_from_keras_model(
        model,hls_config = config,output_dir = dirname + str(i) + '/reuse_' + str(RF) + '/hls4ml_prj',part = 'xc7z007s-clg225-2')
        hls_model.compile()

#         X_test = np.ascontiguousarray(X_test)
#         y_p = hls_model.predict(X_test)
#         y_pred = to_categorical(np.argmax(y_p,axis=1),5)
#         acc = accuracy_score(y_pred,Y_test)
        acc = compare_test(hls_model,X_test,Y_test)
        with open(dirname + str(i) + "/accuracy.txt",'w') as f:
            f.write(str(acc))
            f.close()
        with open(dirname + str(i) + "/parameters.txt",'w') as f:
            f.write(str(layer_2[i]) + " " + str(layer_3[i]) + " " + str(layer_4[i]) + " " + str(bits[i]) + " " + str(int_bits[i]) + " " + str(sparsity[i]) +  " " + str(precision[i]) + " " + str(selector[i]))
            f.close()
        print(dirname + str(i))
        accuracy.append(acc)
    return accuracy,models
