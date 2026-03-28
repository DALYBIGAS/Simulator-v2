#ifndef __HW_DEFINES_H__
#define __HW_DEFINES_H__

// 硬件寄存器和设备定义
#define DEV_INIT    0x01     // 设备初始化标志
#define DEV_INTR    0x04     // 中断标志

// 定义行列大小
#define row_size    ROW
#define col_size    COL
#define mat_size    row_size * col_size * sizeof(TYPE)  // 矩阵大小

// 小波变换相关参数
#define WAVELET_FILTER_SIZE  2  // 低通和高通滤波器大小（例如 Haar）
#define WAVELET_LEVEL        4  // 小波变换层数
#define INPUT_SIZE           row_size * col_size  // 输入信号大小
#define OUTPUT_SIZE          row_size * col_size  // 输出信号大小

// DMA 寄存器定义
#define DMA_CONTROL_REGISTER 0x40000000  // DMA 控制寄存器地址
#define DMA_STATUS_REGISTER  0x40000004  // DMA 状态寄存器地址
#define DMA_SRC_ADDR         0x40000008  // DMA 源地址寄存器
#define DMA_DEST_ADDR        0x4000000C  // DMA 目的地址寄存器
#define DMA_SIZE             0x40000010  // DMA 传输大小寄存器

// DMA 控制信号
#define DMA_CTRL_START       0x1  // 启动 DMA 传输
#define DMA_STATUS_DONE      0x2  // DMA 传输完成标志

// 设备控制和状态寄存器
#define CTRL_REG_ADDRESS     0x50000000  // 控制寄存器
#define STATUS_REG_ADDRESS   0x50000004  // 状态寄存器

// 打开调试模式
#define DEBUG_MODE           1  // 1 = 打开调试，0 = 关闭调试

#endif // __HW_DEFINES_H__
