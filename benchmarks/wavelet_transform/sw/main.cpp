#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "bench.h"
#include "../../../common/m5ops.h"
#include "../wavelet_clstr_hw_defines.h"

wavelet_struct ws;  // 小波变换数据结构

volatile uint8_t  * top   = (uint8_t  *)(TOP + 0x00);
volatile uint32_t * val_a = (uint32_t *)(TOP + 0x01);
volatile uint32_t * val_b = (uint32_t *)(TOP + 0x09);
volatile uint32_t * val_c = (uint32_t *)(TOP + 0x11);

int __attribute__ ((optimize("0"))) main(void) {
    m5_reset_stats();
    uint32_t base = 0x80c00000;
    TYPE *input_signal = (TYPE *)base;
    TYPE *output_signal = (TYPE *)(base + 8 * ROW * COL);
    TYPE *check = (TYPE *)(base + 16 * ROW * COL);
    int row_size = ROW;
    int col_size = COL;
    volatile int count = 0;
    stage = 0;

    // 初始化小波变换结构
    ws.input_signal = input_signal;
    ws.output_signal = output_signal;
    ws.row_size = row_size;
    ws.col_size = col_size;

    printf("Generating data\n");
    genData(&ws);  // 生成输入信号数据
    printf("Data generated\n");

    // 将输入信号地址和输出信号地址传递给硬件
    *val_a = (uint32_t)(void *)input_signal;
    *val_b = (uint32_t)(void *)output_signal;
    *val_c = (uint32_t)(void *)check;

    // 启动硬件加速的小波变换
    *top = 0x01;
    while (stage < 1) count++;  // 等待硬件加速完成

    printf("Job complete\n");

#ifdef CHECK
    // 如果启用了检查，进行结果验证
    printf("Checking result\n");
    printf("Running bench on CPU\n");
    bool fail = false;
    int i, j;
    TYPE sum = 0;
    TYPE mult = 0;

    // 在 CPU 上运行基准测试，计算小波变换结果（可选）
    for (i = 0; i < ROW; i++) {
        for (j = 0; j < COL; j++) {
            sum = input_signal[i * COL + j] * 2;  // 示例，实际小波变换处理
            check[i * COL + j] = sum;
        }
    }

    // 比较 CPU 计算与硬件加速计算的结果
    printf("Comparing CPU run to accelerated run\n");
    for (i = 0; i < ROW * COL; i++) {
        if (output_signal[i] != check[i]) {
            printf("Expected: %f Actual: %f\n", check[i], output_signal[i]);
            fail = true;
            break;
        }
    }

    if (fail)
        printf("Check Failed\n");
    else
        printf("Check Passed\n");
#endif

    m5_dump_stats();
    m5_exit();
}
