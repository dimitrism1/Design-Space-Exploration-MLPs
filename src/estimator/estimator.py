import os
import numpy as np
import pandas as pd
import tensorflow as tf
import hls4ml
import matplotlib.pyplot as plt
import math
from statistics import mean
import joblib
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import qkeras.qtools.qtools_util

####int_bits means weight int_bits. Precision is the total bits for the layer
class estimator():
    def __init__(self,model,precision=8,int_bits=0,reuse=1,layer=0,DSP_mul=1,input_int_bits=0):#,layer_int_bits = 0):
    	
    	self.it_min = -128
    	self.it_max = 128
    	self.sign_bits = 1
    	self.input_int_bits=input_int_bits
    	self.int_bits = int_bits
    	self.DSP_mul = round(DSP_mul)
    	self.layer = layer
    	self.RF = reuse
    	self.loaded_model = model
    	self.precision = precision
    	self.weights = self.loaded_model.weights[layer]
    	qkeras.utils.get_model_sparsity(self.loaded_model)
    	self.new_w = []
    	self.alr = 0
    	for w in self.weights:
            seen = {}
            w = w.numpy()
            for i,weight in enumerate(w):
                if float(weight) in seen:
                    w[i] = 0
                    if float(weight) != 0:
                    	self.alr += 1
                else:
                	seen[float(weight)] = True
            self.new_w.append(w)

    	self.tot_zeros = 0
    	for elem in self.new_w:
            self.tot_zeros += len(elem) - np.count_nonzero(elem)
    	self.zeros = 0
    	for elem in self.weights:
            self.zeros += len(elem) - np.count_nonzero(elem)
    	seen_elem = self.tot_zeros - self.zeros
    	
    	muls = self.loaded_model.weights[self.layer].shape[0] * self.loaded_model.weights[self.layer].shape[1]
    	self.real_muls = math.ceil((muls - self.zeros)/self.RF)
    	self.rm = muls - self.zeros
    	self.tot_real_muls = math.ceil((muls - self.tot_zeros)/self.RF)
    
    def estim_LUT(self,suppress = False):
        precision = self.precision
        layer = self.layer
        loaded_model = self.loaded_model
        RF = self.RF
        int_bits = self.int_bits
        sign_bits = self.sign_bits
        LUT_file = []
        with open("./estimator/saved_luts_sim/mult_" + str(precision) + ".txt",'r') as f:
            for line in f:
                LUT_file.append(int(line.split("|")[1].replace("\n","")))
        LUT_file = np.array(LUT_file)

        LUT_file_DSP = []
        with open("./estimator/saved_luts_DSP/mult_" + str(precision) + ".txt",'r') as f:
                for line in f:
                	LUT_file_DSP.append(int(line.split("|")[1].replace("\n","")))
        LUT_file_DSP = np.array(LUT_file_DSP)
        LUT_orig = []
        with open("./estimator/saved_luts_sim/mult_8.txt") as w:
        	for line in w:
        		LUT_orig.append(int(line.split("|")[1].replace("\n","")))
	

        exp = 8 - int_bits - sign_bits

        norm_weights = [x*2**exp for x in self.new_w]


        DSP_mul = self.DSP_mul
        mul_luts = 0
        mul_ins = 0
        real_bias = 0
        zer = 0
        LUT_impl = 0
        real_bias = np.count_nonzero(loaded_model.weights[layer+1])
        
        
        for w in norm_weights:
                for weight in w:
                    if DSP_mul:
                        if LUT_orig[int(weight) + 128] != 62:
                            mul_luts += LUT_file_DSP[int(weight) + 128]
                            if LUT_file_DSP[int(weight) + 128] == 0 and weight != 0:
                            	zer+=1
                        else:
                        
                        	if precision == 2:
                        		mul_ins += 1
                        		if self.rm < 60:
                        			mul_luts+= 16
                        		elif self.rm < 100:
                        			mul_luts += 17
                        		elif self.rm < 250:
                        			mul_luts += 17
                	       		elif self.rm < 350:
                	       			mul_luts += 14
                	       		elif self.rm < 500:
                	       			mul_luts += 14
                	       		elif self.rm < 1000:
                        			mul_luts += 14
                        		else:
                        			mul_luts += 14
				
                        
                        
                        	elif precision ==3:
                        		mul_ins += 1
                        		if self.rm < 60:
                        			mul_luts+= 17
                        		elif self.rm < 100:
                        			mul_luts += 17
                        		elif self.rm < 250:
                        			mul_luts += 17
                	       		elif self.rm < 350:
                	       			mul_luts += 14
                	       		elif self.rm < 500:
                	       			mul_luts += 13
                	       		elif self.rm < 1000:
                        			mul_luts += 13
                        		else:
                        			mul_luts += 14
				
                        	elif precision == 4:
                        		mul_ins += 1
                        		if self.rm < 60:
                        			mul_luts+= 17
                        		elif self.rm < 100:
                        			mul_luts += 17
                        		elif self.rm < 250:
                        			mul_luts += 17
                	       		elif self.rm < 350:
                	       			mul_luts += 15
                	       		elif self.rm < 500:
                	       			mul_luts += 14
                	       		elif self.rm < 1000:
                        			mul_luts += 14
                        		else:
                        			mul_luts += 14
				
			
                        	elif precision == 5:
                        		mul_ins += 1
                        		if self.rm < 60:
                        			mul_luts+= 17
                        		elif self.rm < 100:
                        			mul_luts += 16
                        		elif self.rm < 250:
                        			mul_luts += 16
                	       		elif self.rm < 350:
                	       			mul_luts += 14
                	       		elif self.rm < 500:
                	       			mul_luts += 15
                	       		elif self.rm < 1000:
                        			mul_luts += 16
                        		else:
                        			mul_luts += 16
			
			
                        	elif precision == 6:
                        		mul_ins += 1
                        		if self.rm < 60:
                        			mul_luts+= 17
                        		elif self.rm < 100:
                        			mul_luts += 17
                        		elif self.rm < 250:
                        			mul_luts += 17
                	       		elif self.rm < 350:
                	       			mul_luts += 16
                	       		elif self.rm < 500:
                	       			mul_luts += 17
                	       		elif self.rm < 1000:
                        			mul_luts += 16
                        		else:
                        			mul_luts += 14
					
			
			
                        	elif precision == 7:
                        		mul_ins += 1
                        		if self.rm < 60:
                        			mul_luts+= 17
                        		elif self.rm < 100:
                        			mul_luts += 16
                        		elif self.rm < 250:
                        			mul_luts += 17
                	       		elif self.rm < 350:
                	       			mul_luts += 14
                	       		elif self.rm < 500:
                	       			mul_luts += 17
                	       		elif self.rm < 1000:
                        			mul_luts += 17
                        		else:
                        			mul_luts += 16
				
			
                        	else:
                        		mul_ins += 1
                        		if self.rm < 60:
                        			mul_luts+= 17
                        		elif self.rm < 100:
                        			mul_luts += 17
                        		elif self.rm < 250:
                        			mul_luts += 17
                	       		elif self.rm < 350:
                	       			mul_luts += 14
                	       		elif self.rm < 500:
                	       			mul_luts += 16
                	       		elif self.rm < 1000:
                        			mul_luts += 16
                        		else:
                        			mul_luts += 16
				
                		
                    else:
                        if LUT_orig[int(weight) + 128] != 62:
                            mul_luts += LUT_file[int(weight) + 128]
                        else:
                        	mul_ins += 1
                        	LUT_impl += LUT_file[int(weight) + 128]

        if mul_ins > self.real_muls:
                mul_luts += 0 if DSP_mul else math.ceil(self.real_muls*62)
                self.DSP = self.real_muls
        else:
                self.DSP = mul_ins

                mul_luts += 0 if DSP_mul else LUT_impl
        if not self.DSP_mul:
        	self.DSP = 0
        lut_accum = 0
        total_add = (self.rm - loaded_model.weights[layer].shape[1]) + real_bias
        
        if precision == 8:
        	if self.rm < 50:
        		add_multiplier = 13
        	
        	elif self.rm < 150:
        		add_multiplier = 13   #self.accum_bits
        	elif self.rm < 400:
        		add_multiplier = 11
        	else:
        		add_multiplier = 11
        
        
        
        
        if precision == 7:
        	if self.rm < 50:
        		add_multiplier = 13
        	
        	elif self.rm < 150:
        		add_multiplier = 13   #self.accum_bits
        	elif self.rm < 400:
        		add_multiplier = 12
        	else:
        		add_multiplier = 11
        
        elif precision == 6:
        	if self.rm < 50:
        		add_multiplier = 14
        	
        	elif self.rm < 150:
        		add_multiplier = 12   #self.accum_bits
        	elif self.rm < 400:
        		add_multiplier = 11
        	else:
        		add_multiplier = 12
        
        
        elif precision == 5:
        	if self.rm < 50:
        		add_multiplier = 14
        	
        	elif self.rm < 150:
        		add_multiplier = 12   #self.accum_bits
        	elif self.rm < 400:
        		add_multiplier = 11
        	else:
        		add_multiplier = 11
        
       
       
        
        elif precision == 4:
        	if self.rm < 50:
        		add_multiplier = 13
        	elif self.rm < 150:
        		add_multiplier = 12   #self.accum_bits
        	elif self.rm < 400:
        		add_multiplier = 11
        	else:
        		add_multiplier = 12
	
        
        elif precision == 3:
        	if self.rm < 50:
        		add_multiplier = 14
        	elif self.rm < 150:
        		add_multiplier = 12   #self.accum_bits
        	elif self.rm < 400:
        		add_multiplier = 11
        	else:
        		add_multiplier = 11
	
	
        elif precision == 2:
        	if self.rm < 50:
        		add_multiplier = 14
        	elif self.rm < 150:
        		add_multiplier = 12   #self.accum_bits
        	elif self.rm < 400:
        		add_multiplier = 11
        	else:
        		add_multiplier = 12
	
	
        lut_accum = total_add * (add_multiplier)# + self.input_int_bits) 

        total_lut = mul_luts + lut_accum
        lut_filename = "./Regression/models/lut_regression.joblib"
        if(self.RF >= 2):
        	lut_model =  joblib.load(lut_filename)
        	lut_ft = PolynomialFeatures(degree = 2,include_bias = False)
        	lut_ft = lut_ft.fit_transform([[total_lut]])
        	dif = math.ceil(lut_model.predict(lut_ft))
        	total_lut = (total_lut - dif) if (DSP_mul) else total_lut
        if not suppress:
        	print("zero LUT elements: " + str(zer))
        	print("Total LUTs: " + str(total_lut))
        	print("Zeros: " + str(self.zeros))
        	print("Total multiplication LUTs: " + str(mul_luts))
        	print("Total accumulator LUTs: " + str(lut_accum))
        	print("Total DSPs: " +str(self.DSP))
        	print("Total layer additions: " + str(total_add - real_bias))
        	print("Total bias additions: " + str(real_bias))
        	print("Number of repeated weights: " + str(self.alr))
        return total_lut
        
    def estim_FF(self,suppress = False):
    	if(self.RF>5):
    		self.RF = 4
    	


    	poly_ft = PolynomialFeatures(degree=4,include_bias=False)
    	filename = {0:"./Regression/models/multi_poly_lut.joblib",1:"./Regression/models/multi_poly.joblib"}
    	ff_model = joblib.load(filename[self.DSP_mul])
    	mul = self.rm
    	reuse = self.RF
    	prec = self.precision
    	end = pd.DataFrame([[mul,reuse,prec]])
    	end = poly_ft.fit_transform(end)
    	FFs = ff_model.predict(end)


    	

    	if not suppress:
	    	print("Total Flip Flops: " + str(math.ceil(FFs[0])))
	    	print("Real multiplications: " + str(self.rm))
    	return math.ceil(FFs[0])
    
    def estim_DSP(self,suppress = False):
    	return self.DSP
    def estim_resource(self,suppress = False):
    	#self.estim_LUT()
    	#self.estim_FF()
    	print("-----------------------------------------------------------")
    	return self.estim_LUT(suppress), self.estim_FF(suppress),self.DSP
    
    @classmethod
    def estim_model(self,model,precision=np.zeros(1)+8,int_bits=np.ones(1),reuse=1,DSP_mul=np.ones(1),suppress = False,dirname = "",device = "Z7007S"):

        precision = precision.round().astype(int)
        int_bits = int_bits.round().astype(int)
        metric = []
        device_FF = {"Z7007S":28800, "Z7012S":68800, "Z7014S":81200, "Z7010":35200, "Z7015":92400, "Z7020":106400}
        device_LUT = {"Z7007S":14400, "Z7012S":34400, "Z7014S":40600, "Z7010":17600, "Z7015":46200, "Z7020":53200}
        device_DSP = {"Z7007S":66, "Z7012S":120, "Z7014S":170, "Z7010":80, "Z7015":160, "Z7020":220}

        for j in range(len(model)):
            FF_pred = []
            LUT_pred = []
            DSP_pred = []
            com_metric = []
            for i in range(0,len(model[j].layers),2):
                c = estimator(model[j],precision[j],int_bits[j],reuse,i,DSP_mul[j])
                #print(j)
                LUT_pred.append((c.estim_LUT(suppress=True))/device_LUT[device])
                FF_pred.append((c.estim_FF(suppress=True))/device_FF[device])
                DSP_pred.append(c.estim_DSP()/device_DSP[device])
                #com_metric.append(np.array(FF_pred[0]) + np.array(LUT_pred[0]) + np.array(DSP_pred[0]))
                #print(com_metric)
                #print("FF_pred is " + str(FF_pred))
                      
                #print("LUT_pred is " + str(LUT_pred))
                #print("DSP_pred is " + str(DSP_pred))

            metric.append(sum(FF_pred + LUT_pred + DSP_pred))
            if dirname != "":
                if os.path.exists(dirname + str(j) + '/reuse_' + str(reuse)):
                    with open(dirname + str(j) + '/reuse_' + str(reuse) + "/size.txt",'w') as f:
                        f.write(str(metric[j]))
                        f.close()
                else:
                    os.mkdir(dirname + str(j) + '/reuse_' + str(reuse))
                    with open(dirname + str(j) + '/reuse_' + str(reuse) + "/size.txt",'w') as f:
                        f.write(str(metric[j]))
                        f.close()
                        
                        
            #print(metric)
            print("-----------------------------------------------------------")
        #return FF_pred,LUT_pred,DSP_pred
            
        return(metric)



