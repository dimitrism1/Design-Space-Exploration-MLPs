#1 /bin/bash
cd LUT_DSP/prec_$1/it_$2
source /tools/Xilinx/Vitis_HLS/2023.2/settings64.sh
vitis_hls -f ../run_test.tcl 

