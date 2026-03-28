#include <stdio.h>
#include "bench.h"

extern volatile uint8_t * top;

void isr(void) {
    printf("Interrupt\n");
    stage += 1;  // 增加计算阶段
    *top = 0x00;  // 清除中断标志
    printf("Interrupt finished\n");
}
