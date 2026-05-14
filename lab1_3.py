# Task 2: Profiling with PyTorch and Perfetto
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prepare input (batch size=1, context len=1000)
prompt = "Hello! " * 200  # ~1000 tokens for most LLMs
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1000)
inputs = {k: v.to(device) for k, v in inputs.items()}

# Warm-up iterations
print("Running warm-up iterations...")
for _ in range(3):
	_ = model.generate(**inputs, max_new_tokens=20)

# CPU profiling
with torch.profiler.profile(
	activities=[torch.profiler.ProfilerActivity.CPU],
	record_shapes=True,
	profile_memory=True,
	with_stack=True,
	on_trace_ready=torch.profiler.tensorboard_trace_handler("./cpu_profile")
) as prof:
	_ = model.generate(**inputs, max_new_tokens=20)

print("CPU profiling complete. Trace saved to ./cpu_profile.")

# GPU profiling (if available)
if torch.cuda.is_available():
	with torch.profiler.profile(
		activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
		record_shapes=True,
		profile_memory=True,
		with_stack=True,
		on_trace_ready=torch.profiler.tensorboard_trace_handler("./gpu_profile")
	) as prof:
		_ = model.generate(**inputs, max_new_tokens=20)
	print("GPU profiling complete. Trace saved to ./gpu_profile.")
else:
	print("CUDA not available, skipping GPU profiling.")

print("\nOpen the trace files in Perfetto or TensorBoard for detailed analysis.")
