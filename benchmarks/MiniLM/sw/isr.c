#include <stdio.h>

extern volatile uint8_t * top;
extern volatile int stage;

void isr(void)
{
	printf("Interrupt\n");
	stage += 1;
	*top = 0x00;
	// printf("%d\n", *top);
	printf("Interrupt finished\n");
}