// ==============================================================
// Vitis HLS - High-Level Synthesis from C, C++ and OpenCL v2022.1 (64-bit)
// Version: 2022.1
// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// ==============================================================

`timescale 1 ns / 1 ps

(* use_dsp = "yes" *) module myproject_mul_8s_7s_15_3_0(clk, ce, reset, din0, din1, dout);
parameter ID = 1;
parameter NUM_STAGE = 2;
parameter din0_WIDTH = 14;
parameter din1_WIDTH = 12;
parameter dout_WIDTH = 26;
input clk;
input ce;
input reset;
input[din0_WIDTH - 1 : 0] din0; 
input[din1_WIDTH - 1 : 0] din1; 
output[dout_WIDTH - 1 : 0] dout;

reg signed [din0_WIDTH - 1 : 0] a_reg0;
reg signed [din1_WIDTH - 1 : 0] b_reg0;
wire signed [dout_WIDTH - 1 : 0] tmp_product;
reg signed [dout_WIDTH - 1 : 0] buff0;

assign dout = buff0;
assign tmp_product = a_reg0 * b_reg0;
always @ (posedge clk) begin
    if (ce) begin
        a_reg0 <= din0;
        b_reg0 <= din1;
        buff0 <= tmp_product;
    end
end
endmodule
