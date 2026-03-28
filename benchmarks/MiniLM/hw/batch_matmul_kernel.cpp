// 请在该文件中实现加速器算法代码，并用#pragma unroll n原语实现并行
#include "../batch_matmul_kernel_cluster_hw_defines.h"

void batch_matmul_kernel(){
    float    * m1     = (float    *)BATCH_MATMUL_KERNEL_BUF_0;
    float    * m2     = (float    *)BATCH_MATMUL_KERNEL_BUF_1;
    float    * m3     = (float    *)BATCH_MATMUL_KERNEL_BUF_2;
    int k_col, i_col, i_o_col;
    float mult, sum;
    for(int ij=0; ij<2304; ij++) {
        // Column index
        int i = ij % 6;
        // Row index
        int j = ij / 384;

        i_col = i * 384;
        sum = 0;
        #pragma unroll
        for(int k=0;k<384;k++) {
            k_col = k * 384;
            mult = m1[i_col + k] * m2[k_col + j];
            sum += mult;
        }
        i_o_col = i * 384;
        m3[i_o_col + j]  = sum;
    }
}
