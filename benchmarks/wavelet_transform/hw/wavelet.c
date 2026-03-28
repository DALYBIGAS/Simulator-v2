#include "hw_defines.h"

// 小波变换的核心计算
void wavelet_transform() {
    TYPE *input_data = (TYPE *)MATRIX1;   // 输入信号
    TYPE *output_data = (TYPE *)MATRIX3;  // 输出信号
    TYPE temp_data[row_size * col_size];  // 临时存储数据
    int half_size = row_size * col_size / 2;  // 下采样后的大小
    TYPE lowpass[] = {0.5, 0.5};  // 简单的低通滤波器（以 Haar 小波为例）
    TYPE highpass[] = {0.5, -0.5}; // 高通滤波器（Haar）

    // 遍历每个信号的元素，进行小波变换
    for (int ij = 0; ij < row_size * col_size; ij++) {
        // 行索引
        int i = ij % row_size;
        // 列索引
        int j = ij / row_size;

        int i_col = i * col_size;
        TYPE sum_low = 0;
        TYPE sum_high = 0;

        // 应用低通和高通滤波器
        #pragma unroll
        for (int k = 0; k < row_size; k++) {
            int k_col = k * col_size;
            sum_low += input_data[i_col + k] * lowpass[0] + input_data[k_col + j] * lowpass[1];
            sum_high += input_data[i_col + k] * highpass[0] + input_data[k_col + j] * highpass[1];
        }

        // 保存低频部分（低通滤波结果）和高频部分（高通滤波结果）
        temp_data[i_col + j] = sum_low;   // 低频部分
        output_data[half_size + i_col + j] = sum_high;  // 高频部分

    }

    // 进行下采样：保留低频部分并处理
    for (int i = 0; i < half_size; i++) {
        output_data[i] = temp_data[i];
    }

    // 可选：对低频部分进行递归小波变换
    if (half_size > 1) {
        wavelet_transform();
    }
}
