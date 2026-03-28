#ifndef DMA_TRANSFER_H
#define DMA_TRANSFER_H

#include <stdint.h>

void dma_transfer(volatile uint8_t *dma_flags, uint64_t read_addr,
                  uint64_t write_addr, uint32_t copy_len);

void start_stream_dma_transfer(volatile uint8_t *str_dma_flags, uint64_t rd_addr, uint64_t wr_addr,
                               uint32_t rd_frame_size, uint8_t num_rd_frames, uint8_t rd_frame_buff_size,
                               uint32_t wr_frame_size, uint8_t num_wr_frames, uint8_t wr_frame_buff_size);

void poll_stream_dma_transfer(volatile uint8_t *str_dma_flags);

void dma_transfer_tensor_to_spm(volatile uint8_t *dma_flags, uint64_t write_addr,
                                uint64_t data_offset, const uint32_t *shape, const uint32_t *stride,
                                uint32_t copy_len, uint64_t aligned_ptr, uint32_t num_dims);

void dma_transfer_tensor_to_mem(volatile uint8_t *dma_flags, uint64_t read_addr,
                                uint64_t data_offset, const uint32_t *shape, const uint32_t *stride,
                                uint32_t copy_len, uint64_t aligned_ptr, uint32_t num_dims);

#endif // DMA_TRANSFER_H
