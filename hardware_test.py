# test local hardware
from transformers import AutoModelForCausalLM
MODEL = "google/gemma-3-4b-it"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="cuda:0",
    torch_dtype="auto",
    trust_remote_code=True,
)