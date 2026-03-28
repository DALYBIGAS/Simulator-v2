#include <stdint.h>

void accelerator_call(volatile uint8_t *acc_flags) {
    // Start the accelerator
    *acc_flags = 0x01; // DEV_INIT

    // Poll function for finish
    while ((*acc_flags & 0x04) != 0x04); // DEV_INTR
}
