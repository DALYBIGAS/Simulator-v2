#include <stdint.h>
#include <stddef.h>

typedef struct {
  void *data;
  int64_t *shape;
  int64_t *stride;
  int32_t rank;
} ai_tensor_ref;
// Auto-generated host stub for mode=decode, kernel=decode_attention_stage3
void decode_attention_stage3_launch(ai_tensor_ref query, ai_tensor_ref key_cache, ai_tensor_ref value_cache, ai_tensor_ref attn_out, volatile uint8_t *dma_flags, uint64_t accelerator_base) {
  // TODO: Replace placeholder register map with accelerator-specific layout.
  volatile uint64_t *accel_kernel_id = (volatile uint64_t *)(accelerator_base + 0x00);
  volatile uint64_t *accel_arg_base = (volatile uint64_t *)(accelerator_base + 0x40);
  volatile uint64_t *accel_ctrl = (volatile uint64_t *)(accelerator_base + 0x08);

  *accel_kernel_id = 0; // kernel slot for decode_attention_stage3
  accel_arg_base[0] = (uint64_t)query.data;
  accel_arg_base[1] = (uint64_t)query.shape;
  accel_arg_base[2] = (uint64_t)query.stride;
  accel_arg_base[3] = (uint64_t)query.rank;
  accel_arg_base[4] = (uint64_t)key_cache.data;
  accel_arg_base[5] = (uint64_t)key_cache.shape;
  accel_arg_base[6] = (uint64_t)key_cache.stride;
  accel_arg_base[7] = (uint64_t)key_cache.rank;
  accel_arg_base[8] = (uint64_t)value_cache.data;
  accel_arg_base[9] = (uint64_t)value_cache.shape;
  accel_arg_base[10] = (uint64_t)value_cache.stride;
  accel_arg_base[11] = (uint64_t)value_cache.rank;
  accel_arg_base[12] = (uint64_t)attn_out.data;
  accel_arg_base[13] = (uint64_t)attn_out.shape;
  accel_arg_base[14] = (uint64_t)attn_out.stride;
  accel_arg_base[15] = (uint64_t)attn_out.rank;

  (void)dma_flags; // reserved for async DMA orchestration
  *accel_ctrl = 0x1;
  while (((*accel_ctrl) & 0x4) != 0x4) {
    ;
  }
}

// input tensor: query (uint16_t)
// input tensor: key_cache (uint16_t)
// input tensor: value_cache (uint16_t)
// output tensor: attn_out (uint16_t)