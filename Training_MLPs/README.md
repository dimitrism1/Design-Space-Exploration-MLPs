## Regression model preparation


### Model creation
4 different MLPs were trained using qkeras, synthesized and analyzed to create a dataset for regression models for the Flip-Flops.
Qkeras is an extension to keras, which allows the user to quantize the weights, the biases and the activation functions. In this case the quantization was set to 8 bits and 0 for the integer part. So the weights range from -1 to 127/128 with a distance of 1/128 between them.
This process can easily be extended to other classification MLPs, and either create a dataset for training or to test an existing regression model. The training weights can be extended quantized to different weights to create a more diverse dataset and create regression models for each quantization separately.

The number of layers is set for each MLP and the size is set randomly for each iteration. The interval can be modified before training. It is suggested that they are set so that the number of multiplications varies for each MLP and the dataset contains enough multiplications in all intervals to create a decent regression model. In our case the model provides sufficient regression for multiplication in the range 0-2500. It is not advised to use the this regression model for MLPs larger than that, unless further training takes places.

### Model training
The models created, are trained with a random sparsity using tensorflow's pruning_schedule. The sparsity applied by the user might not be reflected in the end result as quantization also tends to set weights to 0 in case they are already small.  

### HLS configuration
The number of models is determined by the user, but it is suggested that it is high enough so that enough layers are present for a satisfactory regression model. If the models are already trained they can be loaded, and the training part can be skipped. Each layer is configured separately. They are set to a specific precision from 2 to 8, and the reuse factor ranges from 1 to 10 and then from 10 to 100 in steps of a at a time. This creates in total model_num*7*19=133*model_num different hls_models where model_num is the number of qkeras models that were trained, in this case 20.

### HLS synthesis
Each HLS model is synthesized to extract its resource utilization. To run synthesis several scripts where used and can be found in the scripts redirectory of the repository. The init.sh bash script is called in the case of generic multiplier implementation on DSPs and init_lut.sh in case of LUT implementation. The script starts the synthesis using the build tcl script. The script extracts the parameters of the run(clock,device,project_name) from the project_orig.tcl script and inserts the multiplier implementation. After each run the results are saved in the dirname directory, which is specifed prior to the run. In the end of each iteration, the autopilot files are removed as they consume unnecessarily much disk space. From the init.sh or init_lut.sh script the user has to change the Vitis version or it won't run synthesis

### Running the training scripts
To train each model simply run the bash script inside each directory
```console
source run.sh 
```
It is advised to first verify the target directory where the results will be saved
