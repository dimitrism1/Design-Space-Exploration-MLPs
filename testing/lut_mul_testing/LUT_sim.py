import os
import numpy as np
import tensorflow as tf
import hls4ml


###### Multiply an x bit number that represents the precision of a layer, with a number between -128 and 127 that represents a 8bit weight. The results are arrays that show how many LUTs are needed for each multiplication when the multiplication implementation is on LUTs

prec_min = 2
prec_max = 8
it_min = -128
it_max = 127
for pr in range(prec_min,prec_max,1):
    for i in range(it_min,it_max,1):
        if not(os.path.exists('./LUT_sim/prec_' + str(pr) + "/it_" + str(i) + '/')):
            os.makedirs('./LUT_sim/prec_' + str(pr) + "/it_" + str(i))
        with open('./LUT_sim/prec_' + str(pr) + "/it_" + str(i) + "/mult.cpp",'w' ) as f:
            f.write("""#include <iostream> 
    #include <ap_int.h> 
    extern "C" { 
    void mult(ap_int<""" + str(pr) + """> bitnum,ap_int<""" + str(pr + 8) + """> *out){ 
    ap_int<""" + str(pr + 8) + """> tmp; 
    tmp =""" + str(i)+""" * bitnum;
    *out = tmp;
    }
    }"""
    )
os.system("mv run_sim.sh ./LUT_sim")
            
for pr in range (prec_min,prec_max,1):
    for i in range(it_min,it_max,1):
        with open('./LUT_sim/prec_' + str(pr) + '/run_test.tcl','w') as w:
            w.write("""set XPART xc7z007s-clg225-2
    set CSIM 0
    set CSYNTH 1
    set COSIM 0
    set VIVADO_SYN 0
    set VIVADO_IMPL 0 
    set PROJ mult.prj
    \nset SOLN solution1
    set CLKP 3.33
    open_project -reset $PROJ""" +
    "\nadd_files ./mult.cpp"
    """\nset_top mult
    open_solution -reset $SOLN
    set_part $XPART
    create_clock -period $CLKP
    config_op mul -impl fabric
    set_clock_uncertainty 0.10


    if {$CSYNTH == 1} {
      csynth_design
    }

    exit"""
    )
        w.close()


        os.system('bash ./LUT_sim/run_sim.sh ' + str(pr) + " " + str(i))