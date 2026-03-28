#include <stdint.h>
#include "dma_transfer.h"

void dma_transfer(volatile uint8_t *dma_flags, uint64_t read_addr,
                  uint64_t write_addr, uint32_t copy_len) {
    volatile uint64_t *dma_rd_addr = (volatile uint64_t *)(dma_flags + 1);
    volatile uint64_t *dma_wr_addr = (volatile uint64_t *)(dma_flags + 9);
    volatile uint32_t *dma_copy_len = (volatile uint32_t *)(dma_flags + 17);

    *dma_rd_addr = read_addr;
    *dma_wr_addr = write_addr;
    *dma_copy_len = copy_len;
    *dma_flags = 0x01;

    while ((*dma_flags & 0x04) != 0x04) {
        ;
    }
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

    *str_dma_rd_addr = rd_addr;
    *str_dma_wr_addr = wr_addr;
    *str_dma_rd_frame_size = rd_frame_size;
    *str_dma_num_rd_frames = num_rd_frames;
    *str_dma_rd_frame_buff_size = rd_frame_buff_size;
    *str_dma_wr_frame_size = wr_frame_size;
    *str_dma_num_wr_frames = num_wr_frames;
    *str_dma_wr_frame_buff_size = wr_frame_buff_size;
    *str_dma_flags = 0x03;
}

void poll_stream_dma_transfer(volatile uint8_t *str_dma_flags) {
    while ((*str_dma_flags & 0x08) == 0x08) {
        ;
    }
}

static void dma_tensor_walk_to_spm(volatile uint8_t *dma_flags,
                                   uint64_t write_addr,
                                   uint64_t aligned_ptr,
                                   const uint32_t *shape,
                                   const uint32_t *stride,
                                   uint32_t num_dims,
                                   uint32_t dim,
                                   uint64_t src_offset,
                                   uint64_t dst_offset,
                                   uint32_t copy_len) {
    if (num_dims == 0) {
        return;
    }
    if (dim + 1 == num_dims) {
        dma_transfer(dma_flags, aligned_ptr + src_offset, write_addr + dst_offset, copy_len);
        return;
    }
    for (uint32_t i = 0; i < shape[dim]; ++i) {
        dma_tensor_walk_to_spm(dma_flags,
                               write_addr,
                               aligned_ptr,
                               shape,
                               stride,
                               num_dims,
                               dim + 1,
                               src_offset + ((uint64_t)i * stride[dim]),
                               dst_offset + ((uint64_t)i * shape[num_dims - 1] * copy_len),
                               copy_len);
    }
}

static void dma_tensor_walk_to_mem(volatile uint8_t *dma_flags,
                                   uint64_t read_addr,
                                   uint64_t aligned_ptr,
                                   const uint32_t *shape,
                                   const uint32_t *stride,
                                   uint32_t num_dims,
                                   uint32_t dim,
                                   uint64_t src_offset,
                                   uint64_t dst_offset,
                                   uint32_t copy_len) {
    if (num_dims == 0) {
        return;
    }
    if (dim + 1 == num_dims) {
        dma_transfer(dma_flags, read_addr + src_offset, aligned_ptr + dst_offset, copy_len);
        return;
    }
    for (uint32_t i = 0; i < shape[dim]; ++i) {
        dma_tensor_walk_to_mem(dma_flags,
                               read_addr,
                               aligned_ptr,
                               shape,
                               stride,
                               num_dims,
                               dim + 1,
                               src_offset + ((uint64_t)i * shape[num_dims - 1] * copy_len),
                               dst_offset + ((uint64_t)i * stride[dim]),
                               copy_len);
    }
}

void dma_transfer_tensor_to_spm(volatile uint8_t *dma_flags, uint64_t write_addr,
                                uint64_t data_offset, const uint32_t *shape, const uint32_t *stride,
                                uint32_t copy_len, uint64_t aligned_ptr, uint32_t num_dims) {
    dma_tensor_walk_to_spm(dma_flags,
                           write_addr,
                           aligned_ptr,
                           shape,
                           stride,
                           num_dims,
                           0,
                           data_offset,
                           0,
                           copy_len);
}

void dma_transfer_tensor_to_mem(volatile uint8_t *dma_flags, uint64_t read_addr,
                                uint64_t data_offset, const uint32_t *shape, const uint32_t *stride,
                                uint32_t copy_len, uint64_t aligned_ptr, uint32_t num_dims) {
    dma_tensor_walk_to_mem(dma_flags,
                           read_addr,
                           aligned_ptr,
                           shape,
                           stride,
                           num_dims,
                           0,
                           0,
                           data_offset,
                           copy_len);
}
