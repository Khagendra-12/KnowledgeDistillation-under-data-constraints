import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

student = AutoModelForCausalLM.from_pretrained(
    "Locutusque/TinyMistral-248M",
    device_map="auto",
    quantization_config=bnb_config
)

print("student loaded")