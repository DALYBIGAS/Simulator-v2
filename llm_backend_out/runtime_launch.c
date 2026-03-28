#include <stdint.h>
#include <stddef.h>

typedef struct {
  void *data;
  const int64_t *shape;
  const int64_t *stride;
  int32_t rank;
} ai_tensor_ref;

typedef struct {
  uint32_t grid[3];
  uint32_t block[3];
  uint32_t stream_id;
} ai_launch_desc;

static void ai_wait_event(const char *name) { (void)name; }
static void ai_emit_event(const char *name) { (void)name; }
static void ai_launch_kernel(uint64_t accelerator_base, const char *kernel_name, const ai_launch_desc *desc) {
  (void)accelerator_base;
  (void)kernel_name;
  (void)desc;
}

void run_compiled_kernel(uint64_t accelerator_base) {
  // Auto-generated runtime launch plan.
  // buffer[0] name=hidden_states dtype=bf16 rank=3 space=sram double_buffered=1
  // buffer[1] name=w_qkv dtype=bf16 rank=2 space=global double_buffered=0
  // buffer[2] name=qkv_out dtype=bf16 rank=3 space=global double_buffered=0

  ai_wait_event("dma-ready");
  {
    ai_launch_desc desc = {
      .grid = {16, 1, 32},
      .block = {128, 128, 64},
      .stream_id = 0,
    };
    ai_launch_kernel(accelerator_base, "llama_prefill", &desc);
  }
}
