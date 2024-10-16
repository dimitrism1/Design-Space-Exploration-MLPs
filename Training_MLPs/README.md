## Regression model preparation


### Model creation
4 different MLPs were trained using qkeras, synthesized and analyzed to create a dataset for regression models for the Flip-Flops
qkeras is an extension to keras, which allows the user to quantize the weights. In this case the quantization was set to 8 bits and 0 for the integer part. So the weights range from -1 to 127/128 with a distance of 1/128 between them.
This process can easily be extended to other classification MLPs, and either create a dataset for training or to test an existing regression model

The number of layers is set for each MLP and the size is set randomly for each iteration. The interval can be modified before training. It is suggested that they are set so that the number of multiplications varies for each MLP and the dataset contains enough multiplications in all intervals to create a decent regression model

### Model training
