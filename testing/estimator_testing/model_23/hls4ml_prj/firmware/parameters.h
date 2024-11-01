#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include "ap_fixed.h"
#include "ap_int.h"

#include "nnet_utils/nnet_code_gen.h"
#include "nnet_utils/nnet_helpers.h"
// hls-fpga-machine-learning insert includes
#include "nnet_utils/nnet_activation.h"
#include "nnet_utils/nnet_activation_stream.h"
#include "nnet_utils/nnet_dense.h"
#include "nnet_utils/nnet_dense_compressed.h"
#include "nnet_utils/nnet_dense_stream.h"

// hls-fpga-machine-learning insert weights
#include "weights/w2.h"
#include "weights/b2.h"
#include "weights/w5.h"
#include "weights/b5.h"
#include "weights/w8.h"
#include "weights/b8.h"

// hls-fpga-machine-learning insert layer-config
// fc1
struct config2 : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 9;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 43;
    static const unsigned n_nonzeros = 38;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias2_t bias_t;
    typedef weight2_t weight_t;
    typedef layer2_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// relu1
struct relu_config4 : nnet::activ_config {
    static const unsigned n_in = 9;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef relu1_table_t table_t;
};

// fc2
struct config5 : nnet::dense_config {
    static const unsigned n_in = 9;
    static const unsigned n_out = 31;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 149;
    static const unsigned n_nonzeros = 130;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias5_t bias_t;
    typedef weight5_t weight_t;
    typedef layer5_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// relu2
struct relu_config7 : nnet::activ_config {
    static const unsigned n_in = 31;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    typedef relu2_table_t table_t;
};

// fc3
struct config8 : nnet::dense_config {
    static const unsigned n_in = 31;
    static const unsigned n_out = 39;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned strategy = nnet::latency;
    static const unsigned reuse_factor = 1;
    static const unsigned n_zeros = 649;
    static const unsigned n_nonzeros = 560;
    static const unsigned multiplier_limit = DIV_ROUNDUP(n_in * n_out, reuse_factor) - n_zeros / reuse_factor;
    static const bool store_weights_in_bram = false;
    typedef model_default_t accum_t;
    typedef bias8_t bias_t;
    typedef weight8_t weight_t;
    typedef layer8_index index_t;
    template<class x_T, class y_T>
    using product = nnet::product::mult<x_T, y_T>;
};

// softmax
struct softmax_config10 : nnet::activ_config {
    static const unsigned n_in = 39;
    static const unsigned table_size = 1024;
    static const unsigned io_type = nnet::io_parallel;
    static const unsigned reuse_factor = 1;
    static const unsigned axis = -1;
    static const nnet::softmax_implementation implementation = nnet::softmax_implementation::stable;
    typedef softmax_exp_table_t exp_table_t;
    typedef softmax_inv_table_t inv_table_t;
};


#endif
