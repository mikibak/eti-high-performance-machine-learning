from transformers import AutoModelForCausalLM
import torch

model_name = "Qwen/Qwen3-4B-Thinking-2507"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters (PyTorch): {total_params:,}")

# Print all parameter names and sizes
print("\nParameter breakdown:")
for name, param in model.named_parameters():
	print(f"{name}: {param.numel():,}")
