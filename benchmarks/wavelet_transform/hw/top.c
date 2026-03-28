#include "hw_defines.h"
#include <stdint.h>
void top(uint64_t m1_addr,
         uint64_t m2_addr,
         uint64_t m3_addr) {

    // 定义设备 MMRs
    volatile uint8_t  * GEMMFlags  = (uint8_t *)GEMM;
    volatile uint8_t  * DmaFlags   = (uint8_t  *)(DMA_Flags);
    volatile uint64_t * DmaRdAddr  = (uint64_t *)(DMA_RdAddr);
    volatile uint64_t * DmaWrAddr  = (uint64_t *)(DMA_WrAddr);
    volatile uint32_t * DmaCopyLen = (uint32_t *)(DMA_CopyLen);

    // 传输输入数据
    // 传输 M1（输入信号）
    *DmaRdAddr  = m1_addr;
    *DmaWrAddr  = MATRIX1;
    *DmaCopyLen = mat_size;
    *DmaFlags   = DEV_INIT;
    // 等待 DMA 完成
    while ((*DmaFlags & DEV_INTR) != DEV_INTR);

    // 传输 M2（滤波器或小波变换参数）
    *DmaRdAddr  = m2_addr;
    *DmaWrAddr  = MATRIX2;
    *DmaCopyLen = mat_size;  // 假设 M2 是滤波器矩阵
    *DmaFlags   = DEV_INIT;
    // 等待 DMA 完成
    while ((*DmaFlags & DEV_INTR) != DEV_INTR);

    // 启动加速器进行小波变换
    *GEMMFlags = DEV_INIT;
    // 等待加速器完成小波变换
    while ((*GEMMFlags & DEV_INTR) != DEV_INTR);

    // 传输结果 M3（小波变换结果）
    *DmaRdAddr  = MATRIX3;
    *DmaWrAddr  = m3_addr;
    *DmaCopyLen = mat_size;  // 传输结果矩阵
    *DmaFlags   = DEV_INIT;
    // 等待 DMA 完成
    while ((*DmaFlags & DEV_INTR) != DEV_INTR);

    return;
}
