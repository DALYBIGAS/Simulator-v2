# 该文件是输入程序的模板文件，
# 请提供<model>和<example_input>的定义以生成model对应的MLIR文件。

import torch
import torch_mlir

# -----------------------------------

# 请提供<model>和<example_input>变量定义

# -----------------------------------

linalg_on_tensors_mlir = torch_mlir.compile(
    model,
    example_input,
    output_type="LINALG_ON_TENSORS",
    use_tracing=True)

mlir_file_path = '02-linalg-on-tensors.mlir'
with open(mlir_file_path, 'wt') as f:
    print(linalg_on_tensors_mlir.operation.get_asm(), file=f)

result = model.forward(example_input)
print(result)
