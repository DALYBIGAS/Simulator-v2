import torch
import torch_mlir
import transformers
from transformers import MobileViTFeatureExtractor, MobileViTForImageClassification
from PIL import Image
import requests

print("Transformers version:", transformers.__version__)

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

feature_extractor = MobileViTFeatureExtractor.from_pretrained("apple/mobilevit-xx-small")
model = MobileViTForImageClassification.from_pretrained("apple/mobilevit-xx-small")

inputs = feature_extractor(images=image, return_tensors="pt")

outputs = model(**inputs)
logits = outputs.logits

# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

example_input = inputs
model = model.eval()

class OnlyLogitsHuggingFaceModel(torch.nn.Module):
    """Wrapper that returns only the logits from a HuggingFace model."""

    def __init__(self, model_name: str):
        super().__init__()
        self.model = MobileViTForImageClassification.from_pretrained(model_name)
        self.model.eval()

    def forward(self, input):
        return self.model(input).logits

model_name = "apple/mobilevit-small"
model = OnlyLogitsHuggingFaceModel(model_name)

linalg_on_tensors_mlir = torch_mlir.compile(
    model,
    example_input['pixel_values'],
    output_type="LINALG_ON_TENSORS",
    use_tracing=True)

mlir_file_path = 'mobilevit-xx-small-linalg-on-tensors.mlir'
with open(mlir_file_path, 'wt') as f:
    print(linalg_on_tensors_mlir.operation.get_asm(), file=f)

result = model.forward(example_input['pixel_values'])
print(result)
