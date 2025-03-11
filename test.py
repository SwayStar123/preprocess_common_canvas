# pip install accelerate
import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq
checkpoint = "HuggingFaceTB/SmolVLM-Base"
tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir="../../models/SmolVLM")
# for fp16 use `torch_dtype=torch.float16` instead
model = AutoModelForVision2Seq.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16, cache_dir="../../models/SmolVLM")

padded_len = 32

inputs = []
for prompt in ["I am a smol VLM", "I am a capable VLM"]:
    input = tokenizer.encode(prompt, return_tensors="pt", padding=True).to("cuda").squeeze()
    if input.shape[0] < padded_len:
        padding = torch.zeros(padded_len - input.shape[0], dtype=input.dtype, device=input.device)
        input = torch.cat([input, padding])
    inputs.append(input)

inputs = torch.stack(inputs)
outputs = model(inputs, output_hidden_states=True).hidden_states
print(inputs.shape)
print(outputs[-1].size())
# print(tokenizer.decode(outputs[0]))
