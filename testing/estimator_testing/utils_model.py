import os
import numpy as np
import tensorflow as tf
import hls4ml
import estimator as est1
from tensorflow.keras.models import load_model
from qkeras.utils import _add_supported_quantized_objects
import qkeras.qtools.qtools_util
import qkeras.estimate

class utils():
    @classmethod    
    def get_resources(self,filename,reuse="",model=""):
        FF = []
        LUT = []
        if reuse == "":
        	select = 0
        else:
        	select = 1
        with open(filename + "/reuse_"*select + str(reuse) + "/hls4ml_prj/myproject_prj/solution1/syn/report/csynth.rpt") as f:
            for i,lines in enumerate(f):
                if(i == 22):
                    d = lines.split("|")
                    if len(d) > 1:
                        if(d[12].replace(" ","")!='-'):
                            FF.append(int(d[12].replace(" ","").split("(")[0]))
                    	
                        else:
            	            FF.append(0)
                        if(d[13].replace(" ","")!="-"):
                        	LUT.append(int(d[13].replace(" ","").split("(")[0]))
                        else:
                        	LUT.append(0)
                if(i==24):
                    d = lines.split("|")
                    if len(d) > 1:
                        if(d[12].replace(" ","")!='-'):
                            FF.append(int(d[12].replace(" ","").split("(")[0]))
                        else:
                    	    FF.append(0)
                        LUT.append(int(d[13].replace(" ","").split("(")[0]))
                if(i == 26):
                    d = lines.split("|")
                    if len(d) > 1:
                        if(d[12].replace(" ","")!='-'):
                            FF.append(int(d[12].replace(" ","").split("(")[0]))
                        else:
                    	    FF.append(0)
                        LUT.append(int(d[13].replace(" ","").split("(")[0]))
                if(i==28):
                    d = lines.split("|")
                    if len(d) > 1:
                        if(d[12].replace(" ","")!='-'):
                            FF.append(int(d[12].replace(" ","").split("(")[0]))
                        

                        else:
                    	    FF.append(0)
                        LUT.append(int(d[13].replace(" ","").split("(")[0]))
                    break
        return FF,LUT
    
    
    @classmethod    
    def gather(self,loaded_model,prec = 8,reuse = 1,DSP = True,accum_bits=16,accum_int_bits=6,suppress = False):
        LUT_pred = []
        FF_pred = []
        obj = []
        self.model = loaded_model
        for j in range(0,int(len(loaded_model.layers)),2):
            int_part = 0
            obj.append(est1.estimator(precision=prec,model=loaded_model,int_bits=int_part,layer=j,reuse=reuse,DSP_mul=DSP,accum_bits=accum_bits,accum_int_bits=accum_int_bits))
            LUT,FF = obj[int(j/2)].estim_resource(suppress)
            LUT_pred.append(LUT) 
            FF_pred.append(FF)#(obj[int(j/2)].estim_resource())
        return LUT_pred,FF_pred
    
    @classmethod    

    def compare(self,real_LUT,real_FF,estim_LUT,estim_FF,filename,RF):

        if not os.path.exists(filename) :
                with open(filename,'w') as f:
                    f.write("FF_real | FF_pred | FF_error | LUT_real | LUT_pred | LUT error\n")
                #else:
                    f.close()
        else:
            if os.stat(filename).st_size ==0:
                with open(filename,'w') as f:
                    f.write("FF_real | FF_pred | FF_error | LUT_real | LUT_pred | LUT error\n")
                #else:
                    f.close()
        with open(filename,'a') as f:
            for i,LUT in enumerate (real_LUT):
            	dif = 0
            	for j in range(len(real_LUT[0])):
            		
        #f.write(str(1))
                	lut_error = round(100*abs(LUT[j] - estim_LUT[i][j])/LUT[j],3)
                	ff_error = round(100*abs(real_FF[i][j] - estim_FF[i][j])/real_FF[i][j],3)
                	if int(estim_FF[i][j]) < 0:
                		dif += real_FF[i][j]
                		estim_FF[i][j] = 0
                	f.write(str(real_FF[i][j]) + "   | " + str(estim_FF[i][j]) + "   | " + str(ff_error) + "   | " + str(LUT[j]) + "    | " + str(estim_LUT[i][j]) + "    | " + str(lut_error) +    "|" + "    RF=" + str(RF[i]) + "\n")
            	total_real_LUT=np.sum(LUT)
            	total_real_FF=np.sum(real_FF[i]) - dif
            	total_estim_LUT=np.sum(estim_LUT[i])
            	total_estim_FF=np.sum(estim_FF[i])
            	total_lut_error=round(100*abs(total_real_LUT - total_estim_LUT)/total_real_LUT,3)
            	total_ff_error = round(100*abs(total_real_FF - total_estim_FF)/total_real_FF,3)
            	f.write(str(total_real_FF) + "   | " + str(total_estim_FF) + "   | " + str(total_ff_error) + "   | " + str(total_real_LUT) + "    | " + str(total_estim_LUT) + "    | " + str(total_lut_error) + "     Total\n")
            	f.write("-------------------------------------------------------------\n")
    
    @classmethod    
    
    def clear_file(self,filename):
        os.system("rm " + filename)
    
    
    
    @classmethod
    def write_to_files(self,Neural_name,reuse,precision_min=2,precision_max=8,destination = 'results'):
        for precision in range(precision_min,precision_max + 1,1):
            NN_name =Neural_name # "Breast_cancer"
            if NN_name[0].islower() and NN_name != 'jet_tagging':
                 NN_name = NN_name[0].upper() + NN_name[1:]
            id_num = {'jet_tagging':1,'Arrythmia':1,'Breast_cancer':0,"Cardio":0}
            model_path = "../" + NN_name + "/models/models_" + str(id_num[NN_name]) + "/prec_8"# + "/model_"#11/"
            resource_path = "../" + NN_name + "/models/models_" + str(id_num[NN_name]) + "/prec_" + str(precision)
            dirname = "./results/" + NN_name + "/prec_" + str(precision)
            txt_name = dirname  + "/reuse_" + str(reuse) + "_" + destination + ".txt"
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            self.clear_file(txt_name)
            model_num = 20
            for i in range(model_num):
                co = {}
                _add_supported_quantized_objects(co)
                model = load_model(model_path + "/model_" + str(i) + '/KERAS_check_best_model.h5', custom_objects=co)
                qkeras.estimate.get_model_sparsity(model)
                FF,LUT = self.get_resources(resource_path + "/model_" + str(i),reuse)
                LUT_pred,FF_pred = self.gather(precision,model,reuse,True,16,6)
                self.compare(LUT,FF,LUT_pred,FF_pred,txt_name)
