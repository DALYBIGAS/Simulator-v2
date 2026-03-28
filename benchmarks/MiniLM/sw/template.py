# 该文件是输入程序的模板文件，
# 请提供<model>和<example_input>的定义以生成model对应的MLIR文件。

import torch
import torch_mlir

# -----------------------------------

# 请提供<model>和<example_input>变量定义

# -----------------------------------

import warnings
warnings.simplefilter("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 指定要从HuggingFace模型托管网站拉取的模型名称
model_name = "philschmid/MiniLM-L6-H384-uncased-sst2"

# -----------------------------------

# 以下为example_input变量定义部分

# 指定要从模型中提取的Tokenizer
def prepare_sentence_tokens(hf_model: str, sentence: str):

    tokenizer = AutoTokenizer.from_pretrained(hf_model)
    # print(tokenizer)

    return torch.tensor([tokenizer.encode(sentence)])

# 提供一段句子作为输入测试样例
sentence = "The quick brown fox jumps over the lazy dog."

# 将该句子利用从模型中捕获到的Tokenizer进行处理，继而得到模型输入
example_input = prepare_sentence_tokens(model_name, sentence)

# -----------------------------------

# 以下为model变量定义部分

# 通过wrapper调整模型特征及参数
class OnlyLogitsHuggingFaceModel(torch.nn.Module):

    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            # 预训练模型名称
            model_name, 
            # 输出标签数
            num_labels=2,
            # 其他配置
            output_attentions=False,
            output_hidden_states=False,
            torchscript=True,
        )
        self.model.eval()

    def forward(self, input):
        # Return only the logits.
        return self.model(input)[0]

model = OnlyLogitsHuggingFaceModel(model_name)
# print(model)

# 至此两个所需变量的定义完成

# -----------------------------------

with open('example_input.log', 'w', encoding='utf-8') as file:
    # 将变量的内容写入文件
    file.write(str(example_input))

linalg_on_tensors_mlir = torch_mlir.compile(
    model,
    example_input,
    output_type="LINALG_ON_TENSORS",
    use_tracing=True)

mlir_file_path = '02-linalg-on-tensors.mlir'
with open(mlir_file_path, 'wt') as f:
    print(linalg_on_tensors_mlir.operation.get_asm(), file=f)

result = model.forward(example_input)
# print(result)
