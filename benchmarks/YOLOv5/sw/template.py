import torch
import torch_mlir

# Model
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Image
im = "https://ultralytics.com/images/zidane.jpg"

# Inference
results = model(im)

example_input = im

linalg_on_tensors_mlir = torch_mlir.compile(
    model,
    example_input,
    output_type="LINALG_ON_TENSORS",
    use_tracing=True)

mlir_file_path = '02-linalg-on-tensors.mlir'
with open(mlir_file_path, 'wt') as f:
    print(linalg_on_tensors_mlir.operation.get_asm(), file=f)

result = model(example_input)
result.print()
