
/*
 * ======================================================================
 * top.cpp
 * ======================================================================
 * This file includes the dma and accelerator scheduling performed by a 
 * virtual top controller.
 *
*/

//Standard Libraries
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdint.h>
#include "../batch_matmul_kernel_cluster_hw_defines.h"

#define DEV_INIT	0x01
#define DEV_INTR	0x04


void top(
		uint32_t i0_aligned,
		uint32_t i0_offset,
		uint32_t i0_size_0,
		uint32_t i0_size_1,
		uint32_t i0_size_2,
		uint32_t i0_stride_0,
		uint32_t i0_stride_1,
		uint32_t i0_stride_2,
		uint32_t i1_aligned,
		uint32_t i1_offset,
		uint32_t i1_size_0,
		uint32_t i1_size_1,
		uint32_t i1_size_2,
		uint32_t i1_stride_0,
		uint32_t i1_stride_1,
		uint32_t i1_stride_2,
		uint32_t o2_aligned,
		uint32_t o2_offset,
		uint32_t o2_size_0,
		uint32_t o2_size_1,
		uint32_t o2_size_2,
		uint32_t o2_stride_0,
		uint32_t o2_stride_1,
		uint32_t o2_stride_2
) {

	//Define Device MMRs
	volatile uint8_t  * BatchMatmulKernelFlags  = (uint8_t *)BATCH_MATMUL_KERNEL;
	volatile uint8_t  * DmaFlags   = (uint8_t  *)(DMA_Flags);
	volatile uint32_t * DmaRdAddr  = (uint32_t *)(DMA_RdAddr);
	volatile uint32_t * DmaWrAddr  = (uint32_t *)(DMA_WrAddr);
	volatile uint32_t * DmaCopyLen = (uint32_t *)(DMA_CopyLen);
    
	// Generating data address and transfering inputs...
	uint32_t i0_accel_addr = BATCH_MATMUL_KERNEL_BUF_0;
	uint32_t i0_addr_dim_0 = i0_aligned + 4 * i0_offset;
	uint32_t i0_addr_dim_1 = i0_addr_dim_0;
	uint32_t i0_addr_dim_2 = i0_addr_dim_1;

	for(int dim_0 = 0; dim_0 < i0_size_0; dim_0++){
		for(int dim_1 = 0; dim_1 < i0_size_1; dim_1++){
			// Transfering Input Tensors...
			// Transfer i0...
			*DmaRdAddr  = i0_addr_dim_2;
			*DmaWrAddr  = i0_accel_addr;
			*DmaCopyLen = 4 * i0_size_2;
			*DmaFlags   = DEV_INIT;
    		// Poll DMA for finish
			while ((*DmaFlags & DEV_INTR) != DEV_INTR) 
			;
			i0_accel_addr += 4 * i0_size_2;
			i0_addr_dim_1 += 4 * i0_stride_1;
			i0_addr_dim_2 = i0_addr_dim_1;
		}
		i0_addr_dim_0 += 4 * i0_stride_0;
		i0_addr_dim_1 = i0_addr_dim_0;
		i0_addr_dim_2 = i0_addr_dim_1;
	}
	uint32_t i1_accel_addr = BATCH_MATMUL_KERNEL_BUF_1;
	uint32_t i1_addr_dim_0 = i1_aligned + 4 * i1_offset;
	uint32_t i1_addr_dim_1 = i1_addr_dim_0;
	uint32_t i1_addr_dim_2 = i1_addr_dim_1;

	for(int dim_0 = 0; dim_0 < i1_size_0; dim_0++){
		for(int dim_1 = 0; dim_1 < i1_size_1; dim_1++){
			// Transfering Input Tensors...
			// Transfer i1...
			*DmaRdAddr  = i1_addr_dim_2;
			*DmaWrAddr  = i1_accel_addr;
			*DmaCopyLen = 4 * i1_size_2;
			*DmaFlags   = DEV_INIT;
			// Poll DMA for finish
			while ((*DmaFlags & DEV_INTR) != DEV_INTR) 
			;	
			i1_accel_addr += 4 * i1_size_2;
			i1_addr_dim_1 += 4 * i1_stride_1;
			i1_addr_dim_2 = i1_addr_dim_1;
		}
		i1_addr_dim_0 += 4 * i1_stride_0;
		i1_addr_dim_1 = i1_addr_dim_0;
		i1_addr_dim_2 = i1_addr_dim_1;
	}
	// Start the accelerator batch_matmul_kernel
	*BatchMatmulKernelFlags = DEV_INIT;
	// Polling for finish...
	while ((*BatchMatmulKernelFlags & DEV_INTR) != DEV_INTR)
	;
	uint32_t o2_accel_addr = BATCH_MATMUL_KERNEL_BUF_2;
	uint32_t o2_addr_dim_0 = o2_aligned + 4 * o2_offset;
	uint32_t o2_addr_dim_1 = o2_addr_dim_0;
	uint32_t o2_addr_dim_2 = o2_addr_dim_1;

	for(int dim_0 = 0; dim_0 < o2_size_0; dim_0++){
		for(int dim_1 = 0; dim_1 < o2_size_1; dim_1++){

			// Transfering output tensors...
			// Transfering o2...
			*DmaRdAddr  = o2_accel_addr;
			*DmaWrAddr  = o2_addr_dim_2;
			*DmaCopyLen = 4 * o2_size_2;
			*DmaFlags   = DEV_INIT;
			// Poll DMA for finish
			while ((*DmaFlags & DEV_INTR) != DEV_INTR) 
			;
			o2_accel_addr += 4 * o2_size_2;
			o2_addr_dim_1 += 4 * o2_stride_1;
			o2_addr_dim_2 = o2_addr_dim_1;
		}
		o2_addr_dim_0 += 4 * o2_stride_0;
		o2_addr_dim_1 = o2_addr_dim_0;
		o2_addr_dim_2 = o2_addr_dim_1;
	}
}


