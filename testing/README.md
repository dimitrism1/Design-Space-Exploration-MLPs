## Testing for LUTs

To find out how many LUTs a multplication needs the following simulation was run. A weight ranging from -128 to 127 was multiplied with an x-bit number where x is the precision of the layer, and ranging from 2 to 8. This was run once when the generic multipliers are implemented on LUTs(LUT_sim notebook) and once for when the mutlipliers were implemented on DSPs(LUT_DSP notebook).

The simulations can be run either through the respective notebooks where the details can be easily configured or through the python scripts. 

The process is this

-HLS multiplication file is created for all weights from -128 to 127
-A tcl script is created that describes how the multiplications are implemented(DSPs or LUTs) and the clock
-The respective bash script is copied into the directory
-The bash script is executed with the tcl configuration
