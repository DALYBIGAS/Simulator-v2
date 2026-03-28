#include "../defines.h"

#define rcIndex     (r * ROW + c)

volatile int stage;

typedef struct {
    TYPE *input_signal;  // 输入信号
    TYPE *output_signal; // 输出信号
    int row_size;
    int col_size;
} wavelet_struct;

// 用于生成小波变换输入信号的数据
void genData(wavelet_struct *ws) {
    int r, c;

    for (r = 0; r < ws->row_size; r++) {
        for (c = 0; c < ws->col_size; c++) {
            ws->input_signal[rcIndex] = (TYPE)(rcIndex);  // 生成简单的测试数据
        }
    }
}
