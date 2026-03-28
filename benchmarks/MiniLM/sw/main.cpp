
/*
 * ======================================================================
 * main.cpp
 * ======================================================================
 * This file includes the interfaces to call gem5-MLIR system.
 *
*/

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../common/m5ops.h"
#include "../batch_matmul_kernel_cluster_hw_defines.h"

volatile int stage;

volatile uint8_t  * top   = (uint8_t  *)(TOP + 0x00);
volatile uint32_t * mmr_i0_aligned = (uint32_t *)(TOP + 1);
volatile uint32_t * mmr_i0_size_0 = (uint32_t *)(TOP + 9);
volatile uint32_t * mmr_i0_size_1 = (uint32_t *)(TOP + 17);
volatile uint32_t * mmr_i0_size_2 = (uint32_t *)(TOP + 25);
volatile uint32_t * mmr_i0_stride_0 = (uint32_t *)(TOP + 33);
volatile uint32_t * mmr_i0_stride_1 = (uint32_t *)(TOP + 41);
volatile uint32_t * mmr_i0_stride_2 = (uint32_t *)(TOP + 49);
volatile uint32_t * mmr_i0_offset = (uint32_t *)(TOP + 57);
volatile uint32_t * mmr_i1_aligned = (uint32_t *)(TOP + 65);
volatile uint32_t * mmr_i1_size_0 = (uint32_t *)(TOP + 73);
volatile uint32_t * mmr_i1_size_1 = (uint32_t *)(TOP + 81);
volatile uint32_t * mmr_i1_size_2 = (uint32_t *)(TOP + 89);
volatile uint32_t * mmr_i1_stride_0 = (uint32_t *)(TOP + 97);
volatile uint32_t * mmr_i1_stride_1 = (uint32_t *)(TOP + 105);
volatile uint32_t * mmr_i1_stride_2 = (uint32_t *)(TOP + 113);
volatile uint32_t * mmr_i1_offset = (uint32_t *)(TOP + 121);
volatile uint32_t * mmr_o2_aligned = (uint32_t *)(TOP + 129);
volatile uint32_t * mmr_o2_size_0 = (uint32_t *)(TOP + 137);
volatile uint32_t * mmr_o2_size_1 = (uint32_t *)(TOP + 145);
volatile uint32_t * mmr_o2_size_2 = (uint32_t *)(TOP + 153);
volatile uint32_t * mmr_o2_stride_0 = (uint32_t *)(TOP + 161);
volatile uint32_t * mmr_o2_stride_1 = (uint32_t *)(TOP + 169);
volatile uint32_t * mmr_o2_stride_2 = (uint32_t *)(TOP + 177);
volatile uint32_t * mmr_o2_offset = (uint32_t *)(TOP + 185);

extern "C" void __attribute__ ((optimize("0")))batch_matmul_kernel(float* i0_allocated, float* i0_aligned, int64_t i0_offset, int64_t i0_size0, int64_t i0_size1, int64_t i0_size2, int64_t i0_stride0, int64_t i0_stride1, int64_t i0_stride2, float* i1_allocated, float* i1_aligned, int64_t i1_offset, int64_t i1_size0, int64_t i1_size1, int64_t i1_size2, int64_t i1_stride0, int64_t i1_stride1, int64_t i1_stride2, float* o2_allocated, float* o2_aligned, int64_t o2_offset, int64_t o2_size0, int64_t o2_size1, int64_t o2_size2, int64_t o2_stride0, int64_t o2_stride1, int64_t o2_stride2) {


    volatile int count = 0;
    stage = 0;
    
    *mmr_i0_aligned = (uint32_t)(void *)i0_aligned;
    *mmr_i0_size_0 = (int64_t)i0_size0;
    *mmr_i0_size_1 = (int64_t)i0_size1;
    *mmr_i0_size_2 = (int64_t)i0_size2;
    *mmr_i0_stride_0 = (int64_t)i0_stride0;
    *mmr_i0_stride_1 = (int64_t)i0_stride1;
    *mmr_i0_stride_2 = (int64_t)i0_stride2;
    *mmr_i0_offset = (int64_t)i0_offset;
    *mmr_i1_aligned = (uint32_t)(void *)i1_aligned;
    *mmr_i1_size_0 = (int64_t)i1_size0;
    *mmr_i1_size_1 = (int64_t)i1_size1;
    *mmr_i1_size_2 = (int64_t)i1_size2;
    *mmr_i1_stride_0 = (int64_t)i1_stride0;
    *mmr_i1_stride_1 = (int64_t)i1_stride1;
    *mmr_i1_stride_2 = (int64_t)i1_stride2;
    *mmr_i1_offset = (int64_t)i1_offset;
    *mmr_o2_aligned = (uint32_t)(void *)o2_aligned;
    *mmr_o2_size_0 = (int64_t)o2_size0;
    *mmr_o2_size_1 = (int64_t)o2_size1;
    *mmr_o2_size_2 = (int64_t)o2_size2;
    *mmr_o2_stride_0 = (int64_t)o2_stride0;
    *mmr_o2_stride_1 = (int64_t)o2_stride1;
    *mmr_o2_stride_2 = (int64_t)o2_stride2;
    *mmr_o2_offset = (int64_t)o2_offset;
    *top = 0x01;
    while (stage < 1) count++;
}


/////////////////////////////// Adding the extern function prototype and main()
/////////////////////////////// function (including the input) by yourself.

// 模型forward函数声明，需用extern关键字指明作用域
extern "C" {
  void *forward(int64_t* a_allocated, int64_t* a_aligned, int64_t a_offset, 
                int64_t a_size0, int64_t a_size1, 
                int64_t a_stride0, int64_t a_stride1, 
                float* b_allocated, float* b_aligned, int64_t b_offset, 
                int64_t b_size0, int64_t b_size1, 
                int64_t b_stride0, int64_t b_stride1);
}

// 主函数定义，在其中分配两个数组分别用于输入和输出：
// 1）输入数组内容与example_input变量打印的张量内容对应；
// 2）输出数组大小与模型定义输出格式对应：如二分类对应float[2]类型数组。
int main(){

  m5_reset_stats();

  // 输入a数组
  int64_t *a = new int64_t[12];
  // 输出b数组
  float *b = new float[2];

  // 输入a数组内容
  a[0] = 101;
  a[1] = 1996;
  a[2] = 4248;
  a[3] = 2829;
  a[4] = 4419;
  a[5] = 14523;
  a[6] = 2058;
  a[7] = 1996;
  a[8] = 13971;
  a[9] = 3899;
  a[10] = 1012;
  a[11] = 102;

  // printf("before call");
  forward(a, a, 0, 1, 12, 1, 1, b, b, 0, 1, 2, 1, 1);
  // printf("after call");

  m5_dump_stats();
  m5_exit();
}

