#include <iostream>
#include <string>
#include <chrono>
#include <time.h>
#include <arm_neon.h>
#include <dispatch/dispatch.h>
#include <thread>
#include <mach/mach_time.h>
#include <math.h>


using namespace std;


float sigmoid(float x){
    x = 1 / (1 + exp(-1 * x));
    return x;
}


float sigmoid_derivative(float x){
    return exp(-1 * x) / pow(1 + exp(-1* x), 2);
}

float cost(float expected_value, float x){
    return pow(0.5 * (expected_value - x), 2);
}

float cost_derivative(float expected_value, float x){
    return expected_value - x;
}


void feed_forward(float* input_layer, float* middle_layer_1, float* middle_layer_2, float* output_layer, float* raw_output_layer, float* weights, float* biases, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size){
    
    float32x4_t zeros = vdupq_n_f32(0.0f);
    float* bump;
    float* extra_bump;
    
    for(int i = 0; i < middle_layer_1_size; i += 4){
        float32x4_t acc0 = zeros;
        float32x4_t acc1 = zeros;
        float32x4_t acc2 = zeros;
        float32x4_t acc3 = zeros;
        for(int j = 0; j < input_layer_size; j += 4){
            //__builtin_prefetch(input_layer + 4, 0, 1);
            //__builtin_prefetch(weights + 64, 0, 1);
            float32x4_t vi = vld1q_f32(input_layer + j);
            float32x4_t vw0 = vld1q_f32(weights + i * input_layer_size + j);
            float32x4_t vw1 = vld1q_f32(weights + (i + 1) * input_layer_size + j);
            float32x4_t vw2 = vld1q_f32(weights + (i + 2) * input_layer_size + j);
            float32x4_t vw3 = vld1q_f32(weights + (i + 3) * input_layer_size + j);
            acc0 = vfmaq_f32(acc0, vi, vw0);
            acc1 = vfmaq_f32(acc1, vi, vw1);
            acc2 = vfmaq_f32(acc2, vi, vw2);
            acc3 = vfmaq_f32(acc3, vi, vw3);
        }
        float32x4_t output = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        float32x4_t bias = vld1q_f32(biases + i);
        output = vaddq_f32(output, bias);
        output = vmaxq_f32(output, zeros);
        vst1q_f32(middle_layer_1 + i, output);
    }

    
    bump = weights + middle_layer_1_size * input_layer_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        float32x4_t acc0 = zeros;
        float32x4_t acc1 = zeros;
        float32x4_t acc2 = zeros;
        float32x4_t acc3 = zeros;
        for(int j = 0; j < middle_layer_1_size; j += 4){
            //__builtin_prefetch(middle_layer_1 + 4, 0, 1);
            //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + 64, 0, 1);
            extra_bump = bump + j;
            float32x4_t vi = vld1q_f32(middle_layer_1 + j);
            float32x4_t vw0 = vld1q_f32(extra_bump + i * middle_layer_1_size);
            float32x4_t vw1 = vld1q_f32(extra_bump + (i + 1) * middle_layer_1_size);
            float32x4_t vw2 = vld1q_f32(extra_bump + (i + 2) * middle_layer_1_size);
            float32x4_t vw3 = vld1q_f32(extra_bump + (i + 3) * middle_layer_1_size);
            acc0 = vfmaq_f32(acc0, vi, vw0);
            acc1 = vfmaq_f32(acc1, vi, vw1);
            acc2 = vfmaq_f32(acc2, vi, vw2);
            acc3 = vfmaq_f32(acc3, vi, vw3);
        }
        float32x4_t output = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        float32x4_t bias = vld1q_f32(biases + middle_layer_1_size + i);
        output = vaddq_f32(output, bias);
        output = vmaxq_f32(output, zeros);
        vst1q_f32(middle_layer_2 + i, output);
    }
    
    float32x4_t acc = zeros;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        //__builtin_prefetch(middle_layer_2 + 4, 0, 1);
        //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + 4, 0, 1);
        float32x4_t vi = vld1q_f32(middle_layer_2 + i);
        float32x4_t vw = vld1q_f32(weights + middle_layer_1_size * input_layer_size + middle_layer_2_size * middle_layer_1_size + i);
        acc = vfmaq_f32(acc, vi, vw);
    }
    raw_output_layer[0] = vaddvq_f32(acc);
    raw_output_layer[0] += biases[middle_layer_1_size + middle_layer_2_size];
    output_layer[0] = sigmoid(raw_output_layer[0]);
    
}

void backpropagation(float* input_layer, float* middle_layer_1, float* middle_layer_2, float* output_layer, float* raw_output_layer, float* weights, float* biases, float* weight_derivatives, float* bias_derivatives, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size, int expected_value){
    
    float32x4_t zeros = vdupq_n_f32(0.0f);
    bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1] = -1 * cost_derivative(expected_value, output_layer[0]) * sigmoid_derivative(raw_output_layer[0]);
    float32x4_t vdb = vdupq_n_f32(bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1]);
    
    int bump = input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        float32x4_t va = vld1q_f32(middle_layer_2 + i);
        float32x4_t vdw = vmulq_f32(va, vdb);
        vst1q_f32(weight_derivatives + bump + i, vdw);
    }
    
    //bump = input_layer_size * middle_layer_1_size;
    for(int i = 0; i < middle_layer_2_size; i += 1){
        uint32x4_t condition0 = vcgtq_f32(vld1q_f32(middle_layer_2 + i), zeros);
        float32x4_t vw = vld1q_f32(weights + bump + i);
        float32x4_t vdw = vbslq_f32(condition0, vmulq_f32(vw, vdb), zeros);
        vst1q_f32(bias_derivatives + middle_layer_1_size + i, vdw);
        for(int j = 0; j < middle_layer_1_size; j += 4){
            //uint32x4_t condition1 = vcgtq_f32(vld1q_f32(middle_layer_1 + j), zeros);
            float32x4_t vw1 = vld1q_f32(weights + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j);
            //load all middle layer 1 activations all at once to avoid repeated reloads
            //vst1q_f32(weight_derivatives + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j, vmulq_f32(va, vdw));
            vst1q_f32(bias_derivatives + j, vfmaq_f32(vld1q_f32(bias_derivatives + j), vw1, vdw));
        }
    }
    
    for(int i = 0; i < middle_layer_1_size; i += 1){
        uint32x4_t condition = vcgtq_f32(vld1q_f32(middle_layer_1 + i), zeros);
        float32x4_t vdb = vld1q_f32(bias_derivatives + i);
        for(int j = 0; j < input_layer_size; j += 4){
            float32x4_t va = vld1q_f32(input_layer + j);
            vst1q_f32(weight_derivatives + i, vbslq_f32(condition, vmulq_f32(va, vdb), zeros));
        }
    }
}

void backpropagation_2(float* input_layer, float* middle_layer_1, float* middle_layer_2, float* output_layer, float* raw_output_layer, float* weights, float* biases, float* weight_derivatives, float* bias_derivatives, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size, int expected_value){
    
    float32x4_t zeros = vdupq_n_f32(0.0f);
    bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1] = -1 * cost_derivative(expected_value, output_layer[0]) * sigmoid_derivative(raw_output_layer[0]);
    float32x4_t vdb0 = vdupq_n_f32(bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1]);
    
    int bump = input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        float32x4_t va = vld1q_f32(middle_layer_2 + i);
        uint32x4_t condition = vcgtq_f32(va, zeros);
        float32x4_t vw = vld1q_f32(weights + bump + i);
        float32x4_t vdw = vmulq_f32(va, vdb0);
        float32x4_t vdb1 = vbslq_f32(condition, vmulq_f32(vw, vdb0), zeros);
        vst1q_f32(weight_derivatives + bump + i, vdw);
        vst1q_f32(bias_derivatives + middle_layer_1_size + i, vdb1);
    }
    
    for(int i = 0; i < middle_layer_2_size; ++i){
        float32x4_t db1 = vdupq_n_f32(bias_derivatives[middle_layer_1_size + i]);
        for(int j = 0; j < middle_layer_1_size; j += 4){
            //load all middle layer 1 activations all at once to avoid repeated reloads
            float32x4_t va = vld1q_f32(middle_layer_1 + j);
            uint32x4_t condition = vcgtq_f32(va, zeros);
            float32x4_t vw = vld1q_f32(weights + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j);
            vst1q_f32(weight_derivatives + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j, vmulq_f32(va, db1));
            vst1q_f32(bias_derivatives + j, vbslq_f32(condition, vfmaq_f32(vld1q_f32(bias_derivatives + j), vw, db1), zeros));
        }
    }
    
    
    for(int i = 0; i < middle_layer_1_size; i += 1){
        float32x4_t vdb = vdupq_n_f32(bias_derivatives[i]);
        for(int j = 0; j < input_layer_size; j += 4){
            float32x4_t va = vld1q_f32(input_layer + j);
            vst1q_f32(weight_derivatives + i * input_layer_size + j, vmulq_f32(va, vdb));
        }
    }
}


void backpropagation_3(float* input_layer, float* middle_layer_1, float* middle_layer_2, float* output_layer, float* raw_output_layer, float* weights, float* biases, float* weight_derivatives, float* bias_derivatives, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size, int expected_value){
    
    float32x4_t zeros = vdupq_n_f32(0.0f);
    bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1] = -1 * cost_derivative(expected_value, output_layer[0]) * sigmoid_derivative(raw_output_layer[0]);
    float32x4_t vdb0 = vdupq_n_f32(bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1]);
    
    int bump = input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        float32x4_t va = vld1q_f32(middle_layer_2 + i);
        vst1q_f32(weight_derivatives + bump + i, vmulq_f32(va, vdb0));
        vst1q_f32(bias_derivatives + middle_layer_1_size + i, vbslq_f32(vcgtq_f32(va, zeros), vmulq_f32(vld1q_f32(weights + bump + i), vdb0), zeros););
    }
       
    for(int i = 0; i < middle_layer_2_size; ++i){
        float32x4_t db1 = vdupq_n_f32(bias_derivatives[middle_layer_1_size + i]);
        for(int j = 0; j < middle_layer_1_size; j += 4){
            //load all middle layer 1 activations all at once to avoid repeated reloads
            float32x4_t va = vld1q_f32(middle_layer_1 + j);
            uint32x4_t condition = vcgtq_f32(va, zeros);
            float32x4_t vw = vld1q_f32(weights + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j);
            vst1q_f32(weight_derivatives + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j, vmulq_f32(va, db1));
            vst1q_f32(bias_derivatives + j, vbslq_f32(condition, vfmaq_f32(vld1q_f32(bias_derivatives + j), vw, db1), zeros));
        }
    }
    
    for(int i = 0; i < middle_layer_1_size; i += 1){
        float32x4_t vdb = vdupq_n_f32(bias_derivatives[i]);
        for(int j = 0; j < input_layer_size; j += 4){
            float32x4_t va = vld1q_f32(input_layer + j);
            vst1q_f32(weight_derivatives + i * input_layer_size + j, vmulq_f32(va, vdb));
        }
    }
}

void backpropagation_4(float* input_layer, float* middle_layer_1, float* middle_layer_2, float* output_layer, float* raw_output_layer, float* weights, float* biases, float* weight_derivatives, float* bias_derivatives, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size, int expected_value){
    
    float32x4_t zeros = vdupq_n_f32(0.0f);
    bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1] = -1 * cost_derivative(expected_value, output_layer[0]) * sigmoid_derivative(raw_output_layer[0]);
    float32x4_t vdb0 = vdupq_n_f32(bias_derivatives[middle_layer_1_size + middle_layer_2_size + output_layer_size - 1]);
    
    int bump = input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        float32x4_t va = vld1q_f32(middle_layer_2 + i);
        vst1q_f32(weight_derivatives + bump + i, vmulq_f32(va, vdb0));
        vst1q_f32(bias_derivatives + middle_layer_1_size + i, vbslq_f32(vcgtq_f32(va, zeros), vmulq_f32(vld1q_f32(weights + bump + i), vdb0), zeros););
    }
       
    for(int i = 0; i < middle_layer_2_size; ++i){
        float32x4_t db1 = vdupq_n_f32(bias_derivatives[middle_layer_1_size + i]);
        for(int j = 0; j < middle_layer_1_size; j += 4){
            //load all middle layer 1 activations all at once to avoid repeated reloads
            float32x4_t va = vld1q_f32(middle_layer_1 + j);
            uint32x4_t condition = vcgtq_f32(va, zeros);
            float32x4_t vw = vld1q_f32(weights + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j);
            vst1q_f32(weight_derivatives + input_layer_size * middle_layer_1_size + i * middle_layer_1_size + j, vmulq_f32(va, db1));
            vst1q_f32(bias_derivatives + j, vbslq_f32(condition, vfmaq_f32(vld1q_f32(bias_derivatives + j), vw, db1), zeros));
        }
    }
    
    for(int i = 0; i < middle_layer_1_size; i += 1){
        float32x4_t vdb = vdupq_n_f32(bias_derivatives[i]);
        for(int j = 0; j < input_layer_size; j += 4){
            float32x4_t va = vld1q_f32(input_layer + j);
            vst1q_f32(weight_derivatives + i * input_layer_size + j, vmulq_f32(va, vdb));
        }
    }
}


void backpropagation_assembly(float* input_layer, float* middle_layer_1, float* middle_layer_2, float* output_layer, float* raw_output_layer, float* weights, float* biases, float* weight_derivatives, float* bias_derivatives, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size, int expected_value){
    //asm volatile(
                 //"mov x1, %[weights] \n"
                 //"mov x2, %[middle_layer_2] \n"
                 //"mov x3, %[middle_layer_1] \n"
                 //"mov x4, %[input_layer] \n"
                 //"mov x5, %[weight_derivatives] \n"
                 //"mov x6, %[bias_derivarives] \n"
                 //"dup v0.4s #0 \n"
                 //"str []\n"
                 //);
}


int main(int argc, const char * argv[]) {
    int input_layer_size = 16;
    int middle_layer_1_size = 32;
    int middle_layer_2_size = 16;
    int output_layer_size = 1;
    
    /*
    float input_layer[] = {0.1f, -0.9f, 1.001f, -2.0f};
    float* middle_layer_1 = (float*)malloc(middle_layer_1_size * sizeof(float));
    float* middle_layer_2 = (float*)malloc(middle_layer_2_size * sizeof(float));
    float* output_layer = (float*)malloc(output_layer_size * sizeof(float));
    float* raw_output_layer = (float*)malloc(output_layer_size * sizeof(float));
    float weights[] = {0.1, 0.4, 1.3, 1.4, -0.3, -2.0, -0.1, -1.75, -0.036, -0.8, 0.0, 0.032, 0.175, -0.3, 0.43, -0.48, 1.2, 1.3, 0.28, 0.35, -0.9, -0.5, -0.55, -0.19, 0.29, 0.18, 0.14, -0.47, 0.2, -0.82, -0.09, 0.06, 0.19, -0.1, -0.2, 0.1};
    float biases[] = {-0.1, 0.2, -0.3, -0.4, -0.12, 0.14, 0.29, -0.3, -0.2};
    float* weight_derivatives = (float*)malloc((input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + middle_layer_2_size * output_layer_size) * sizeof(float));
    float* bias_derivatives = (float*)malloc((middle_layer_1_size + middle_layer_2_size + output_layer_size) * sizeof(float));
     */
    
    
    //float* input_layer = (float*)aligned_alloc(32, input_layer_size * sizeof(float));
    //float* middle_layer_1 = (float*)aligned_alloc(64, middle_layer_1_size * sizeof(float));
    //float* middle_layer_2 = (float*)aligned_alloc(32, middle_layer_2_size * sizeof(float));
    float* input_layer = (float*)malloc(input_layer_size * sizeof(float));
    float* middle_layer_1 = (float*)malloc(middle_layer_1_size * sizeof(float));
    float* middle_layer_2 = (float*)malloc(middle_layer_2_size * sizeof(float));
    float* output_layer = (float*)malloc(output_layer_size * sizeof(float));
    float* raw_output_layer = (float*)malloc(output_layer_size * sizeof(float));
    //float* weights = (float*)aligned_alloc(32, (input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + middle_layer_2_size * output_layer_size) * sizeof(float));
    float* weights = (float*)malloc((input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + middle_layer_2_size * output_layer_size) * sizeof(float));
    float* biases = (float*)malloc((middle_layer_1_size + middle_layer_2_size + output_layer_size) * sizeof(float));
    
    float* weight_derivatives = (float*)aligned_alloc(32, (input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + middle_layer_2_size * output_layer_size) * sizeof(float));
    float* bias_derivatives = (float*)malloc((middle_layer_1_size + middle_layer_2_size + output_layer_size) * sizeof(float));
    
        
    for(int i = 0; i < input_layer_size; ++i){
        input_layer[i] = 2.0f;
    }
    for(int i = 0; i < input_layer_size * middle_layer_1_size * middle_layer_2_size + middle_layer_2_size; ++i){
        weights[i] = 3.0f;
    }
    for(int i = 0; i < middle_layer_1_size + middle_layer_2_size + output_layer_size; ++i){
        biases[i] = 4.0f;
    }
    
    auto start = std::chrono::steady_clock::now();
    
    feed_forward(input_layer, middle_layer_1, middle_layer_2, output_layer, raw_output_layer, weights, biases, input_layer_size, middle_layer_1_size, middle_layer_2_size, output_layer_size);
    //backpropagation_4(input_layer, middle_layer_1, middle_layer_2, output_layer, raw_output_layer, weights, biases, weight_derivatives, bias_derivatives, input_layer_size, middle_layer_1_size, middle_layer_2_size, output_layer_size, 1);
    auto end = std::chrono::steady_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Feed Forward duration: " << nanoseconds << " ns" << std::endl;
    
    
    
    return 0;
}
