#ifndef __DEFINES_H__
#define __DEFINES_H__
#include "hw_defines.h"
#ifndef HW_DEFINES_H
#define HW_DEFINES_H
#include <stdint.h> 
#define TYPE float  
#define ROW 256     
#define COL 256    
#define row_size ROW
#define col_size COL
#define MATRIX1 0x10000000
#define MATRIX2 0x20000000
#define MATRIX3 0x30000000
#define GEMM    0x40000000

#define DMA_Flags  0x50000000
#define DMA_RdAddr 0x50000010
#define DMA_WrAddr 0x50000020
#define DMA_CopyLen 0x50000030

#define mat_size (row_size * col_size * sizeof(TYPE))
#endif
// 硬件定义和优化标志
#define CHECK  // 可用于开启检查模式（例如调试信息）

// 数据类型定义，ECG 信号通常使用 float 或 double 精度
#define TYPE double  // 根据需要选择数据类型，double 提供更高精度

// 信号尺寸定义：ECG 信号的长度可以是固定的，例如 1000 或更长
#define ECG_SIGNAL_SIZE 1000  // 假设 ECG 信号的大小是 1000 点，可以根据实际情况调整

// 小波变换的参数
#define WAVELET_TYPE "db4"   // 选择小波基：Daubechies 小波（4阶），你可以根据实际需要更改
#define WAVELET_LEVEL 4      // 小波变换的层数，通常 4 层到 6 层足够用于信号分解

// ECG 信号的采样率（假设是 500Hz，这也是许多 ECG 数据集的标准）
#define ECG_SAMPLING_RATE 500

// 信号去噪和特征提取的参数（例如选择去除某些频段的系数）
#define DENOISE_THRESHOLD 0.1  // 用于去噪的小波系数阈值，可以根据需要调整

// 其他硬件优化相关设置
#define MAX_CORES 4           // 假设硬件有 4 个加速器核心处理小波变换
#define BUFFER_SIZE 256       // 定义 DMA 缓冲区大小

// 用于调试模式下的打印
#ifdef CHECK
    #define DEBUG_PRINT(fmt, ...) printf(fmt, ##__VA_ARGS__)
#else
    #define DEBUG_PRINT(fmt, ...)  // 不进行打印
#endif

#endif
