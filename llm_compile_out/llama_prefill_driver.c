#include <stdint.h>
#include <stddef.h>

typedef struct {
  void *data;
  int64_t *shape;
  int64_t *stride;
  int32_t rank;
} ai_tensor_ref;
// Auto-generated host stub for mode=prefill, kernel=llama_prefill
void llama_prefill_launch(ai_tensor_ref hidden_states, ai_tensor_ref w_qkv, ai_tensor_ref qkv_out, volatile uint8_t *dma_flags, uint64_t accelerator_base) {
  // TODO: Replace placeholder register map with accelerator-specific layout.
  volatile uint64_t *accel_kernel_id = (volatile uint64_t *)(accelerator_base + 0x00);
  volatile uint64_t *accel_arg_base = (volatile uint64_t *)(accelerator_base + 0x40);
  volatile uint64_t *accel_ctrl = (volatile uint64_t *)(accelerator_base + 0x08);

  *accel_kernel_id = 0; // kernel slot for llama_prefill
  accel_arg_base[0] = (uint64_t)hidden_states.data;
  accel_arg_base[1] = (uint64_t)hidden_states.shape;
  accel_arg_base[2] = (uint64_t)hidden_states.stride;
  accel_arg_base[3] = (uint64_t)hidden_states.rank;
  accel_arg_base[4] = (uint64_t)w_qkv.data;
  accel_arg_base[5] = (uint64_t)w_qkv.shape;
  accel_arg_base[6] = (uint64_t)w_qkv.stride;
  accel_arg_base[7] = (uint64_t)w_qkv.rank;
  accel_arg_base[8] = (uint64_t)qkv_out.data;
  accel_arg_base[9] = (uint64_t)qkv_out.shape;
  accel_arg_base[10] = (uint64_t)qkv_out.stride;
  accel_arg_base[11] = (uint64_t)qkv_out.rank;

  (void)dma_flags; // reserved for async DMA orchestration
  *accel_ctrl = 0x1;
  while (((*accel_ctrl) & 0x4) != 0x4) {
    ;
  }
}

// input tensor: hidden_states (uint16_t)
// input tensor: w_qkv (uint16_t)
// output tensor: qkv_out (uint16_t)