#include <stdint.h>

void dma_transfer(volatile uint8_t *dma_flags, uint64_t read_addr, 
                  uint64_t write_addr, uint32_t copy_len) {
    volatile uint64_t *dma_rd_addr = (volatile uint64_t *)(dma_flags + 1);
    volatile uint64_t *dma_wr_addr = (volatile uint64_t *)(dma_flags + 9);
    volatile uint32_t *dma_copy_len = (volatile uint32_t *)(dma_flags + 17);

    // Configure DMA registers
    *dma_rd_addr = read_addr;
    *dma_wr_addr = write_addr;
    *dma_copy_len = copy_len;
    *dma_flags = 0x01; // DEV_INIT

    // Poll DMA for finish
    while ((*dma_flags & 0x04) != 0x04); // DEV_INTR
}

void start_stream_dma_transfer(volatile uint8_t *str_dma_flags, uint64_t rd_addr, uint64_t wr_addr, 
                               uint32_t rd_frame_size, uint8_t num_rd_frames, uint8_t rd_frame_buff_size, 
                               uint32_t wr_frame_size, uint8_t num_wr_frames, uint8_t wr_frame_buff_size) {
    volatile uint64_t *str_dma_rd_addr = (volatile uint64_t *)(str_dma_flags + 4);
    volatile uint64_t *str_dma_wr_addr = (volatile uint64_t *)(str_dma_flags + 12);
    volatile uint32_t *str_dma_rd_frame_size = (volatile uint32_t *)(str_dma_flags + 20);
    volatile uint8_t *str_dma_num_rd_frames = (volatile uint8_t *)(str_dma_flags + 24);
    volatile uint8_t *str_dma_rd_frame_buff_size = (volatile uint8_t *)(str_dma_flags + 25);
    volatile uint32_t *str_dma_wr_frame_size = (volatile uint32_t *)(str_dma_flags + 26);
    volatile uint8_t *str_dma_num_wr_frames = (volatile uint8_t *)(str_dma_flags + 30);
    volatile uint8_t *str_dma_wr_frame_buff_size = (volatile uint8_t *)(str_dma_flags + 31);

    // Configure Stream DMA registers
    *str_dma_rd_addr = rd_addr;
    *str_dma_wr_addr = wr_addr;
    *str_dma_rd_frame_size = rd_frame_size;
    *str_dma_num_rd_frames = num_rd_frames;
    *str_dma_rd_frame_buff_size = rd_frame_buff_size;
    *str_dma_wr_frame_size = wr_frame_size;
    *str_dma_num_wr_frames = num_wr_frames;
    *str_dma_wr_frame_buff_size = wr_frame_buff_size;
    *str_dma_flags = 0x03; // STR_DMA_INIT_RD | STR_DMA_INIT_WR
}

void poll_stream_dma_transfer(volatile uint8_t *str_dma_flags) {
    // Poll Stream DMA for finish
    while ((*str_dma_flags & 0x08) == 0x08); // STR_DMA_WR_RUNNING
}

void dma_transfer_tensor_to_spm(volatile uint8_t *dma_flags, uint64_t write_addr, 
                                uint64_t data_offset, uint32_t *shape, uint32_t *stride, 
                                uint32_t copy_len, uint64_t aligned_ptr) {
    // Determine the number of dimensions
    uint32_t num_dims = 0;
    while (shape[num_dims] != 0 && stride[num_dims] != 0) {
        num_dims++;
    }

    // Compute the total number of transfers needed
    uint32_t total_transfers = 1;
    for (uint32_t i = 0; i < num_dims - 1; ++i) {
        total_transfers *= shape[i];
    }

    uint64_t linear_offset = data_offset;
    uint64_t wr_addr = write_addr;

    // Perform multiple 1-dimensional transfers
    for (uint32_t i = 0; i < total_transfers; ++i) {

        // Call the existing dma_transfer function
        dma_transfer(dma_flags, aligned_ptr + linear_offset, 
            wr_addr, shape[num_dims]);
    
        linear_offset += shape[num_dims - 1];
        wr_addr += shape[num_dims - 1];
    }
}

void dma_transfer_tensor_to_mem(volatile uint8_t *dma_flags, uint64_t read_addr, 
                                uint64_t data_offset, uint32_t *shape, uint32_t *stride, 
                                uint32_t copy_len, uint64_t aligned_ptr) {
    // Determine the number of dimensions
    uint32_t num_dims = 0;
    while (shape[num_dims] != 0 && stride[num_dims] != 0) {
        num_dims++;
    }

    // Compute the total number of transfers needed
    uint32_t total_transfers = 1;
    for (uint32_t i = 0; i < num_dims - 1; ++i) {
        total_transfers *= shape[i];
    }

    uint64_t linear_offset = data_offset;
    uint64_t rd_addr = read_addr;

    // Perform multiple 1-dimensional transfers
    for (uint32_t i = 0; i < total_transfers; ++i) {

        // Call the existing dma_transfer function
        dma_transfer(dma_flags, rd_addr, aligned_ptr + linear_offset, shape[num_dims]);
    
        linear_offset += shape[num_dims - 1];
        rd_addr += shape[num_dims - 1];
    }
}
