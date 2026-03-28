from transformers import MobileViTForImageClassification
import torch

# Load model
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small")
model.eval()

# Dictionary to store input/output shapes and strides
shape_dict = {}

# Forward hook to record shapes and strides
def record_shapes(module, input, output):
    input_shapes = []
    for i in input:
        if isinstance(i, torch.Tensor):
            input_shapes.append(i.shape)
        elif isinstance(i, (tuple, list)):
            input_shapes.extend([x.shape for x in i if isinstance(x, torch.Tensor)])
    
    output_shapes = []
    if isinstance(output, torch.Tensor):
        output_shapes.append(output.shape)
    elif isinstance(output, (tuple, list)):
        output_shapes.extend([x.shape for x in output if isinstance(x, torch.Tensor)])
    elif hasattr(output, 'last_hidden_state'):  # Handle BaseModelOutput
        output_shapes.append(output.last_hidden_state.shape)
    
    # Get stride information if available
    stride = None
    if hasattr(module, 'stride'):
        stride = module.stride
    elif hasattr(module, 'conv') and hasattr(module.conv, 'stride'):  # For blocks with nested convs
        stride = module.conv.stride
    
    shape_dict[module] = {
        'input': input_shapes,
        'output': output_shapes,
        'stride': stride
    }

# Register hooks on all layers
hooks = []
for name, module in model.named_modules():
    hooks.append(module.register_forward_hook(record_shapes))

# Run inference to populate shape_dict
dummy_input = torch.randn(1, 3, 256, 256)
with torch.no_grad():
    try:
        _ = model(dummy_input)
    except Exception as e:
        print(f"Forward pass failed with: {e}")

# Remove hooks
for hook in hooks:
    hook.remove()

# Print combined information
for name, param in model.named_parameters():
    if "weight" in name:
        # Find which module this parameter belongs to
        module = None
        for m in model.modules():
            if hasattr(m, 'weight') and torch.equal(m.weight, param):
                module = m
                break
        
        if module and module in shape_dict:
            print(f"\nParameter: {name}")
            print(f"Weight shape: {param.shape}")
            if shape_dict[module]['input']:
                print(f"Input shapes: {shape_dict[module]['input']}")
            else:
                print("Input shapes: (No tensor inputs recorded)")
            if shape_dict[module]['output']:
                print(f"Output shapes: {shape_dict[module]['output']}")
            else:
                print("Output shapes: (No tensor outputs recorded)")
            if shape_dict[module]['stride'] is not None:
                print(f"Stride: {shape_dict[module]['stride']}")
            else:
                print("Stride: (No stride information available)")
        else:
            print(f"\nParameter: {name}")
            print(f"Weight shape: {param.shape}")
            print("(Input/output shapes and stride not available)")