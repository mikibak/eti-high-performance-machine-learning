import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import os
import intel_extension_for_pytorch as ipex

# Task 1: IPEX INT8 Quantization

device = torch.device("cpu")

# Model selection
model_name = "Qwen/Qwen2.5-0.5B"  # or "Qwen/Qwen3-4B-Thinking-2507"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in float32 for IPEX quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
)

device = torch.device("cpu")
model = model.to(device)
model.eval()

# Quantize model to INT8 with IPEX
print("Quantizing model to INT8 with IPEX...")
model = ipex.optimize(model, dtype=torch.int8, inplace=True)

# Prefill input (~1000 tokens)
prompt_prefill = "Hello! " * 200

inputs_prefill = tokenizer(
    prompt_prefill,
    return_tensors="pt",
    truncation=True,
    max_length=1000,
)

inputs_prefill = {
    k: v.to(device)
    for k, v in inputs_prefill.items()
}

# Decode input (~1 token)
prompt_decode = "Hello!"

inputs_decode = tokenizer(
    prompt_decode,
    return_tensors="pt",
)

inputs_decode = {
    k: v.to(device)
    for k, v in inputs_decode.items()
}

@torch.no_grad()
def forward_fn(**inputs):
    return model(**inputs)

def measure_latency_and_memory(forward_fn, inputs, n_iter=10):
    process = psutil.Process(os.getpid())
    times = []
    max_mem = 0
    for _ in range(n_iter):
        start_mem = process.memory_info().rss
        start = time.time()
        forward_fn(**inputs)
        end = time.time()
        mem = process.memory_info().rss
        max_mem = max(max_mem, mem, start_mem)
        times.append(end - start)
    avg_time = sum(times) / len(times)
    return avg_time, max_mem / (1024 ** 2)  # Return MB


# --- Profiling INT8 quantized model ---
def profile_inference(forward_fn, inputs, label, trace_file=None):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=(torch.profiler.tensorboard_trace_handler(trace_file) if trace_file else None),
    ) as prof:
        forward_fn(**inputs)
    print(f"\nProfiler summary for {label}:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=15))

print("Measuring latency and memory usage (prefill)...")
prefill_latency, prefill_mem = measure_latency_and_memory(forward_fn, inputs_prefill)
print(f"Prefill latency: {prefill_latency:.6f} s, Peak memory: {prefill_mem:.2f} MB")
profile_inference(forward_fn, inputs_prefill, "INT8 prefill", trace_file="./quantized_prefill_trace")


print("Measuring latency and memory usage (decode, 3 steps)...")
decode_times = []
decode_mems = []
for i in range(3):
    decode_latency, decode_mem = measure_latency_and_memory(forward_fn, inputs_decode, n_iter=1)
    decode_times.append(decode_latency)
    decode_mems.append(decode_mem)
    # Profile only the first decode step for summary and trace
    if i == 0:
        profile_inference(forward_fn, inputs_decode, "INT8 decode", trace_file="./quantized_decode_trace")
avg_decode_latency = sum(decode_times) / len(decode_times)
max_decode_mem = max(decode_mems)
print(f"Decode latency (avg 3 steps): {avg_decode_latency:.6f} s, Peak memory: {max_decode_mem:.2f} MB")

print("\nSummary:")
print(f"INT8 Quantized (IPEX) - Prefill: {prefill_latency:.6f} s, {prefill_mem:.2f} MB | Decode (3 steps avg): {avg_decode_latency:.6f} s, {max_decode_mem:.2f} MB")
