# Task 1: Eager vs Compiled Execution
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Thinking-2507"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Prefill input (context len=1000)
prompt_prefill = "Hello! " * 200
inputs_prefill = tokenizer(prompt_prefill, return_tensors="pt", truncation=True, max_length=1000)
inputs_prefill = {k: v.to(device) for k, v in inputs_prefill.items()}

# Decode input (context len=1)
prompt_decode = "Hello!"
inputs_decode = tokenizer(prompt_decode, return_tensors="pt")
inputs_decode = {k: v.to(device) for k, v in inputs_decode.items()}

# Helper to measure average latency
def measure_latency(forward_fn, inputs, n_iter=5):
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iter):
        start = time.time()
        forward_fn(**inputs)
        torch.cuda.synchronize()
        times.append(time.time() - start)
    return sum(times) / len(times)

# Eager mode
print("Running warm-up (eager)...")
for _ in range(3):
    model.generate(**inputs_prefill, max_new_tokens=20)
    model.generate(**inputs_decode, max_new_tokens=20)

def eager_forward(**inputs):
    return model.generate(**inputs, max_new_tokens=20)

# Compiled mode
compiled_model = torch.compile(model)
def compiled_forward(**inputs):
    return compiled_model.generate(**inputs, max_new_tokens=20)

print("Measuring latency...")
prefill_eager = measure_latency(eager_forward, inputs_prefill)
prefill_compiled = measure_latency(compiled_forward, inputs_prefill)
decode_eager = measure_latency(eager_forward, inputs_decode)
decode_compiled = measure_latency(compiled_forward, inputs_decode)

print(f"Prefill avg latency (eager): {prefill_eager:.4f}s")
print(f"Prefill avg latency (compiled): {prefill_compiled:.4f}s")
print(f"Decode avg latency (eager): {decode_eager:.4f}s")
print(f"Decode avg latency (compiled): {decode_compiled:.4f}s")

# Kernel counting helper
def count_kernels(forward_fn, inputs, activities):
    with torch.profiler.profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        forward_fn(**inputs)
    return len([e for e in prof.events() if e.device_type == torch.profiler.ProfilerActivity.CUDA])

if torch.cuda.is_available():
    print("Counting kernels (eager)...")
    eager_prefill_kernels = count_kernels(eager_forward, inputs_prefill, [torch.profiler.ProfilerActivity.CUDA])
    eager_decode_kernels = count_kernels(eager_forward, inputs_decode, [torch.profiler.ProfilerActivity.CUDA])
    print("Counting kernels (compiled)...")
    compiled_prefill_kernels = count_kernels(compiled_forward, inputs_prefill, [torch.profiler.ProfilerActivity.CUDA])
    compiled_decode_kernels = count_kernels(compiled_forward, inputs_decode, [torch.profiler.ProfilerActivity.CUDA])
    print(f"Prefill kernels (eager): {eager_prefill_kernels}")
    print(f"Prefill kernels (compiled): {compiled_prefill_kernels}")
    print(f"Decode kernels (eager): {eager_decode_kernels}")
    print(f"Decode kernels (compiled): {compiled_decode_kernels}")
    print("\nFusion reduces kernel count if compiled < eager. Speedup is usually higher for prefill because the context is longer and more ops can be fused. Decode is more sequential, so less fusion and speedup.")
else:
    print("CUDA not available, skipping kernel count.")
