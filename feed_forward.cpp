
//  main.cpp
//  HFT MLP (int8)
//
//  Created by Samuel Raymond Ard on 6/5/25.
//

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


int8_t fast_sigmoid(int32_t x){
    //x = x >> 3;
    //return 127 / (1 + exp(-x));
    return x / (1 + abs(x));
}


float cost(float expected_value, float x){
    return pow(0.5 * (expected_value - x), 2);
}


void feed_forward(int8_t* input_layer, int8_t* middle_layer_1, int8_t* middle_layer_2, int8_t* output_layer, int8_t* raw_output_layer, int8_t* weights, int8_t* biases, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size){
    
    int8x16_t zeros = vdupq_n_s8(0.0);
    int8_t* bump;
    int8_t* extra_bump;
    
    for(int i = 0; i < middle_layer_1_size; i += 4){
        int8x16_t acc0 = zeros;
        int8x16_t acc1 = zeros;
        int8x16_t acc2 = zeros;
        int8x16_t acc3 = zeros;
        for(int j = 0; j < input_layer_size; j += 16){
            //__builtin_prefetch(input_layer + 4, 0, 1);
            //__builtin_prefetch(weights + 64, 0, 1);
            int8x16_t vi = vld1q_s8(input_layer + j);
            int8x16_t vw0 = vld1q_s8(weights + i * input_layer_size + j);
            int8x16_t vw1 = vld1q_s8(weights + (i + 1) * input_layer_size + j);
            int8x16_t vw2 = vld1q_s8(weights + (i + 2) * input_layer_size + j);
            int8x16_t vw3 = vld1q_s8(weights + (i + 3) * input_layer_size + j);
            acc0 = vfmaq_f32(acc0, vi, vw0);
            acc1 = vfmaq_f32(acc1, vi, vw1);
            acc2 = vfmaq_f32(acc2, vi, vw2);
            acc3 = vfmaq_f32(acc3, vi, vw3);
        }
        float32x4_t output = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        float32x4_t bias = vld1q_s8(biases + i);
        output = vaddq_f32(output, bias);
        output = vmaxq_f32(output, zeros);
        vst1q_s8(middle_layer_1 + i, output);
    }

    bump = weights + middle_layer_1_size * input_layer_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        int8x16_t acc0 = zeros;
        int8x16_t acc1 = zeros;
        int8x16_t acc2 = zeros;
        int8x16_t acc3 = zeros;
        for(int j = 0; j < middle_layer_1_size; j += 16){
            //__builtin_prefetch(middle_layer_1 + 4, 0, 1);
            //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + 64, 0, 1);
            extra_bump = bump + j;
            int8x16_t vi = vld1q_s8(middle_layer_1 + j);
            int8x16_t vw0 = vld1q_s8(extra_bump + i * middle_layer_1_size);
            int8x16_t vw1 = vld1q_s8(extra_bump + (i + 1) * middle_layer_1_size);
            int8x16_t vw2 = vld1q_s8(extra_bump + (i + 2) * middle_layer_1_size);
            int8x16_t vw3 = vld1q_s8(extra_bump + (i + 3) * middle_layer_1_size);
            acc0 = vfmaq_f32(acc0, vi, vw0);
            acc1 = vfmaq_f32(acc1, vi, vw1);
            acc2 = vfmaq_f32(acc2, vi, vw2);
            acc3 = vfmaq_f32(acc3, vi, vw3);
        }
        float32x4_t output = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        float32x4_t bias = vld1q_s8(biases + middle_layer_1_size + i);
        output = vaddq_f32(output, bias);
        output = vmaxq_f32(output, zeros);
        vst1q_s8(middle_layer_2 + i, output);
    }
    
    int8x16_t acc = zeros;
    for(int i = 0; i < middle_layer_2_size; i += 16){
        //__builtin_prefetch(middle_layer_2 + 4, 0, 1);
        //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + 4, 0, 1);
        int8x16_t vi = vld1q_s8(middle_layer_2 + i);
        int8x16_t vw = vld1q_s8(weights + middle_layer_1_size * input_layer_size + middle_layer_2_size * middle_layer_1_size + i);
        acc = vfmaq_f32(acc, vi, vw);
    }
    raw_output_layer[0] = vaddvq_f32(acc);
    raw_output_layer[0] += biases[middle_layer_1_size + middle_layer_2_size];
    output_layer[0] = sigmoid(raw_output_layer[0]);
}


void feed_forward_2(int8_t* input_layer, int8_t* middle_layer_1, int8_t* middle_layer_2, int8_t* output_layer, int8_t* raw_output_layer, int8_t* weights, int8_t* biases, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size){
    
    int8x16_t zeros = vdupq_n_s8(0.0);
    int8_t* bump;
    int8_t* extra_bump;
    
    for(int i = 0; i < middle_layer_1_size; i += 4){
        int32x4_t acc0 = zeros;
        int32x4_t acc1 = zeros;
        int32x4_t acc2 = zeros;
        int32x4_t acc3 = zeros;
        for(int j = 0; j < input_layer_size; j += 16){
            //__builtin_prefetch(input_layer + 4, 0, 1);
            //__builtin_prefetch(weights + 64, 0, 1);
            int8x16_t vi = vmaxq_s8(vld1q_s8(input_layer + j), zeros);
            int8x16_t vw0 = vld1q_s8(weights + i * input_layer_size + j);
            int8x16_t vw1 = vld1q_s8(weights + (i + 1) * input_layer_size + j);
            int8x16_t vw2 = vld1q_s8(weights + (i + 2) * input_layer_size + j);
            int8x16_t vw3 = vld1q_s8(weights + (i + 3) * input_layer_size + j);
            acc0 = vdotq_s32(acc0, vi, vw0);
            acc1 = vdotq_s32(acc1, vi, vw1);
            acc2 = vdotq_s32(acc2, vi, vw2);
            acc3 = vdotq_s32(acc3, vi, vw3);
        }
        float32x4_t output = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        float32x4_t bias = vld1q_s8(biases + i);
        output = vaddq_f32(output, bias);
        //output = vmaxq_f32(output, zeros);
        vst1q_s8(middle_layer_1 + i, output);
    }

    bump = weights + middle_layer_1_size * input_layer_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        int32x4_t acc0 = zeros;
        int32x4_t acc1 = zeros;
        int32x4_t acc2 = zeros;
        int32x4_t acc3 = zeros;
        for(int j = 0; j < middle_layer_1_size; j += 16){
            //__builtin_prefetch(middle_layer_1 + 4, 0, 1);
            //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + 64, 0, 1);
            extra_bump = bump + j;
            int8x16_t vi = vmaxq_s8(vld1q_s8(middle_layer_1 + j), zeros);
            int8x16_t vw0 = vld1q_s8(extra_bump + i * middle_layer_1_size);
            int8x16_t vw1 = vld1q_s8(extra_bump + (i + 1) * middle_layer_1_size);
            int8x16_t vw2 = vld1q_s8(extra_bump + (i + 2) * middle_layer_1_size);
            int8x16_t vw3 = vld1q_s8(extra_bump + (i + 3) * middle_layer_1_size);
            acc0 = vdotq_s32(acc0, vi, vw0);
            acc1 = vdotq_s32(acc1, vi, vw1);
            acc2 = vdotq_s32(acc2, vi, vw2);
            acc3 = vdotq_s32(acc3, vi, vw3);
        }
        float32x4_t output = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        float32x4_t bias = vld1q_s8(biases + middle_layer_1_size + i);
        output = vaddq_f32(output, bias);
        //output = vmaxq_f32(output, zeros);
        vst1q_s8(middle_layer_2 + i, output);
    }
    
    int32x4_t acc = zeros;
    for(int i = 0; i < middle_layer_2_size; i += 16){
        //__builtin_prefetch(middle_layer_2 + 4, 0, 1);
        //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + 4, 0, 1);
        int8x16_t vi = vmaxq_s8(vld1q_s8(middle_layer_2 + i), zeros);
        int8x16_t vw = vld1q_s8(weights + middle_layer_1_size * input_layer_size + middle_layer_2_size * middle_layer_1_size + i);
        acc = vdotq_s32(acc, vi, vw);
    }
    raw_output_layer[0] = vaddvq_f32(acc);
    raw_output_layer[0] += biases[middle_layer_1_size + middle_layer_2_size];
    output_layer[0] = raw_output_layer[0] / (1 + abs(raw_output_layer[0]));
    
}

void feed_forward_3(int8_t* __restrict input_layer, int8_t* __restrict middle_layer_1, int8_t* __restrict middle_layer_2, int8_t* output_layer, int8_t* raw_output_layer, int8_t* __restrict weights, int8_t* biases, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size){
    
    int8x16_t zeros = vdupq_n_s8(0.0);
    int8_t* bump;
    int8_t* extra_bump;
    
    int8x16_t vi0 = vmaxq_s8(vld1q_s8(input_layer), zeros);
    for(int i = 0; i < middle_layer_1_size; i += 16){
        int32x4_t acc0 = vmulq_s8(vi0, vld1q_s8(weights + i * input_layer_size));
        int32x4_t acc1 = vmulq_s8(vi0, vld1q_s8(weights + (i + 1) * input_layer_size));
        int32x4_t acc2 = vmulq_s8(vi0, vld1q_s8(weights + (i + 2) * input_layer_size));
        int32x4_t acc3 = vmulq_s8(vi0, vld1q_s8(weights + (i + 3) * input_layer_size));
        int32x4_t acc4 = vmulq_s8(vi0, vld1q_s8(weights + (i + 4) * input_layer_size));
        int32x4_t acc5 = vmulq_s8(vi0, vld1q_s8(weights + (i + 5) * input_layer_size));
        int32x4_t acc6 = vmulq_s8(vi0, vld1q_s8(weights + (i + 6) * input_layer_size));
        int32x4_t acc7 = vmulq_s8(vi0, vld1q_s8(weights + (i + 7) * input_layer_size));
        int32x4_t acc8 = vmulq_s8(vi0, vld1q_s8(weights + (i + 8) * input_layer_size));
        int32x4_t acc9 = vmulq_s8(vi0, vld1q_s8(weights + (i + 9) * input_layer_size));
        int32x4_t acc10 = vmulq_s8(vi0, vld1q_s8(weights + (i + 10) * input_layer_size));
        int32x4_t acc11 = vmulq_s8(vi0, vld1q_s8(weights + (i + 11) * input_layer_size));
        int32x4_t acc12 = vmulq_s8(vi0, vld1q_s8(weights + (i + 12) * input_layer_size));
        int32x4_t acc13 = vmulq_s8(vi0, vld1q_s8(weights + (i + 13) * input_layer_size));
        int32x4_t acc14 = vmulq_s8(vi0, vld1q_s8(weights + (i + 14) * input_layer_size));
        int32x4_t acc15 = vmulq_s8(vi0, vld1q_s8(weights + (i + 15) * input_layer_size));
        int8x16_t output = {vaddvq_s8(acc0), vaddvq_s8(acc1), vaddvq_s8(acc2), vaddvq_s8(acc3), vaddvq_s8(acc4), vaddvq_s8(acc5), vaddvq_s8(acc6), vaddvq_s8(acc7), vaddvq_s8(acc8), vaddvq_s8(acc9), vaddvq_s8(acc10), vaddvq_s8(acc11), vaddvq_s8(acc12), vaddvq_s8(acc13), vaddvq_s8(acc14), vaddvq_s8(acc15)};
        int8x16_t bias = vld1q_s8(biases + i);
        output = vaddq_s8(output, bias);
        //output = vmaxq_f32(output, zeros);
        vst1q_s8(middle_layer_1 + i, output);
    }

    bump = weights + middle_layer_1_size * input_layer_size;
    for(int i = 0; i < middle_layer_2_size; i += 4){
        int32x4_t acc0 = zeros;
        int32x4_t acc1 = zeros;
        int32x4_t acc2 = zeros;
        int32x4_t acc3 = zeros;
        for(int j = 0; j < middle_layer_1_size; j += 16){
            //__builtin_prefetch(middle_layer_1 + 4, 0, 1);
            //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + 64, 0, 1);
            extra_bump = bump + j;
            int8x16_t vi = vmaxq_s8(vld1q_s8(middle_layer_1 + j), zeros);
            int8x16_t vw0 = vld1q_s8(extra_bump + i * middle_layer_1_size);
            int8x16_t vw1 = vld1q_s8(extra_bump + (i + 1) * middle_layer_1_size);
            int8x16_t vw2 = vld1q_s8(extra_bump + (i + 2) * middle_layer_1_size);
            int8x16_t vw3 = vld1q_s8(extra_bump + (i + 3) * middle_layer_1_size);
            acc0 = vdotq_s32(acc0, vi, vw0);
            acc1 = vdotq_s32(acc1, vi, vw1);
            acc2 = vdotq_s32(acc2, vi, vw2);
            acc3 = vdotq_s32(acc3, vi, vw3);
        }
        float32x4_t output = {vaddvq_f32(acc0), vaddvq_f32(acc1), vaddvq_f32(acc2), vaddvq_f32(acc3)};
        float32x4_t bias = vld1q_s8(biases + middle_layer_1_size + i);
        output = vaddq_f32(output, bias);
        //output = vmaxq_f32(output, zeros);
        vst1q_s8(middle_layer_2 + i, output);
    }
    
    int32x4_t acc = zeros;
    for(int i = 0; i < middle_layer_2_size; i += 16){
        //__builtin_prefetch(middle_layer_2 + 4, 0, 1);
        //__builtin_prefetch(weights + input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + 4, 0, 1);
        int8x16_t vi = vmaxq_s8(vld1q_s8(middle_layer_2 + i), zeros);
        int8x16_t vw = vld1q_s8(weights + middle_layer_1_size * input_layer_size + middle_layer_2_size * middle_layer_1_size + i);
        acc = vdotq_s32(acc, vi, vw);
    }
    raw_output_layer[0] = vaddvq_f32(acc);
    raw_output_layer[0] += biases[middle_layer_1_size + middle_layer_2_size];
    output_layer[0] = raw_output_layer[0] / (1 + abs(raw_output_layer[0]));
    
}


void feed_forward_4(int8_t* input_layer, int8_t* middle_layer_1, int8_t* middle_layer_2, int8_t* output_layer, int8_t* raw_output_layer, int8_t* weights, int8_t* biases, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size){
    
    //int8x16_t zeros = vdupq_n_s8(0.0);
    //int8_t* bump;
    //int8_t* extra_bump;
    
    int8x16_t va = vld1q_s8(input_layer);
    vst1q_s8(middle_layer_1, vld1q_s8(biases));
    vst1q_s8(middle_layer_1 + 16, vld1q_s8(biases + 16));
    
    for(int i = 0; i < middle_layer_1_size; ++i){
        middle_layer_1[i] += vaddvq_s8(vmulq_s8(va, vld1q_s8(weights + i * input_layer_size)));
    }
    
    
    
}

void feed_forward_5(int8_t* input_layer, int8_t* middle_layer_1, int8_t* middle_layer_2, int8_t* output_layer, int8_t* raw_output_layer, int8_t* weights, int8_t* biases, int input_layer_size, int middle_layer_1_size, int middle_layer_2_size, int output_layer_size){
    
    
    asm volatile(
                 "ld1 {v0.16b}, [%[input_layer]] \n"
                 "mov x1, %[biases] \n"
                 "mov x2, %[middle_layer_1] \n"
                 "mov x9, %[weights] \n"
                 "add x2, x2, #16 \n"
                 "ld1 {v4.16b}, [x1] \n"
                 "st1 {v4.16b}, [x2] \n"
                 "mov w5, #0 \n"
                 "mov w6, %w[middle_layer_1_size] \n"
                 "movi v7.16b, #0 \n"
                 "mov w8, #0 \n"
                 
                 "1: \n"
                 "cmp w6, w5 \n"
                 "b.eq 2f \n"
                 "ldr q10, [x9, x8] \n"
                 "mul v10.16b, v10.16b, v0.16b \n"
                 "addv b10, v10.16b \n"
                 "str b10, [x2, x5] \n"
                 "add w8, w8, #16 \n"
                 "add w5, w5, #1 \n"
                 "b 1b \n"
                 
                 //apply ReLU mask somewhere in the middle, use vmax funcion
                 
                 "2: \n"
                 "ld1 {v0.16b}, [x2] \n"
                 "ld1 {v3.16b}, [x1] \n"
                 "add v0.16b, v0.16b, v3.16b \n"
                 "smax v0.16b, v0.16b, v7.16b \n"
                 "st1 {v0.16b}, [x2] \n"
                 "ld1 {v11.16b}, [x2], #16 \n"
                 "ldr q3, [x1, #16] \n"
                 "add v11.16b, v11.16b, v3.16b \n"
                 "smax v11.16b, v11.16b, v7.16b \n"
                 "str q3, [x2, #16] \n"
                 "mov x2, %[middle_layer_2] \n"
                 "ldr q3, [x1, #32] \n"
                 "mov w5, #0 \n"
                 "mov w6, %w[middle_layer_2_size] \n"
                 
                 "3: \n"
                 "cmp w6, w5 \n"
                 "b.eq 4f \n"
                 "ldr q10, [x9, x8] \n"
                 "add x8, x8, #16 \n"
                 "ldr q12, [x9, x8] \n"
                 "mul v10.16b, v0.16b, v10.16b \n"
                 "mul v12.16b, v11.16b, v12.16b \n"
                 "add v10.16b, v10.16b, v12.16b \n"
                 "addv b0, v10.16b \n"
                 "str b0, [x2, x5] \n"
                 "add w8, w8, #16 \n"
                 "add w5, w5, #1 \n"
                 "b 3b \n"
                 
                 "4: \n"
                 "ld1 {v0.16b}, [x2] \n"
                 "ldr q3, [x1, #32] \n"
                 "add v0.16b, v0.16b, v3.16b \n"
                 "smax v0.16b, v0.16b, v7.16b \n"
                 "st1 {v0.16b}, [x2] \n"
                 "mov x2, %[output_layer] \n"
                 "ldr b1, [x1, #48] \n"
                 "addv b0, v0.16b \n"
                 "add v0.16b, v0.16b, v1.16b \n"
                 //add sigmoid function
                 
                 
                 :[input_layer]"+r"(input_layer), [biases]"+r"(biases), [middle_layer_1]"+r"(middle_layer_1), [middle_layer_1_size]"+r"(middle_layer_1_size), [input_layer_size]"+r"(input_layer_size), [weights]"+r"(weights), [middle_layer_2]"+r"(middle_layer_2), [middle_layer_2_size]"+r"(middle_layer_2_size), [output_layer]"+r"(output_layer)
                 :
                 :"v0", "x1", "x2", "v3", "v4", "w5", "w6", "v7", "w8", "x9", "v10", "cc", "memory"
                 );
    
    
}


int main(int argc, const char * argv[]) {
    int input_layer_size = 16;
    int middle_layer_1_size = 32;
    int middle_layer_2_size = 16;
    int output_layer_size = 1;
    
    int8_t* input_layer = (int8_t*)aligned_alloc(8, input_layer_size * sizeof(int8_t));
    int8_t* middle_layer_1 = (int8_t*)aligned_alloc(8, middle_layer_1_size * sizeof(int8_t));
    int8_t* middle_layer_2 = (int8_t*)aligned_alloc(8, middle_layer_2_size * sizeof(int8_t));
    int8_t* output_layer = (int8_t*)malloc(output_layer_size * sizeof(int8_t));
    int8_t* raw_output_layer = (int8_t*)malloc(output_layer_size * sizeof(int8_t));
    int8_t* weights = (int8_t*)aligned_alloc(8, (input_layer_size * middle_layer_1_size + middle_layer_1_size * middle_layer_2_size + middle_layer_2_size * output_layer_size) * sizeof(int8_t));
    int8_t* biases = (int8_t*)malloc((middle_layer_1_size + middle_layer_2_size + output_layer_size) * sizeof(int8_t));
    
    int8x16_t* temp_layer = (int8x16_t*)calloc(middle_layer_2_size, sizeof(int8x16_t));
    
    for(int i = 0; i < input_layer_size; ++i){
        input_layer[i] = 2.0;
    }
    for(int i = 0; i < input_layer_size * middle_layer_1_size * middle_layer_2_size + middle_layer_2_size; ++i){
        weights[i] = 3.0;
    }
    for(int i = 0; i < middle_layer_1_size + middle_layer_2_size + output_layer_size; ++i){
        biases[i] = 4.0;
    }
    
    auto start = std::chrono::steady_clock::now();
    feed_forward_5(input_layer, middle_layer_1, middle_layer_2, output_layer, raw_output_layer, weights, biases, input_layer_size, middle_layer_1_size, middle_layer_2_size, output_layer_size);
    auto end = std::chrono::steady_clock::now();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Feed Forward 1 duration: " << nanoseconds << " ns" << std::endl;
    
    start = chrono::steady_clock::now();
    //feed_forward_3(input_layer, middle_layer_1, middle_layer_2, output_layer, raw_output_layer, weights, biases, input_layer_size, middle_layer_1_size, middle_layer_2_size, output_layer_size);
    end = chrono::steady_clock::now();
    nanoseconds = chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    std::cout << "Feed Forward 3 duration: " << nanoseconds << " ns" << std::endl;
    
    
    return 0;
}
