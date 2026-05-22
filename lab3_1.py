import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import os

# =============================
# Task 0: FP16 Baseline on CPU
# =============================

model_name = "Qwen/Qwen2.5-0.5B"

torch.set_float32_matmul_precision("high")

torch._dynamo.config.suppress_errors = True

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

device = torch.device("cpu")

model = model.to(device)
model.eval()

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

print("Measuring latency and memory usage (prefill)...")
prefill_latency, prefill_mem = measure_latency_and_memory(forward_fn, inputs_prefill)
print(f"Prefill latency: {prefill_latency:.6f} s, Peak memory: {prefill_mem:.2f} MB")

print("Measuring latency and memory usage (decode)...")
decode_latency, decode_mem = measure_latency_and_memory(forward_fn, inputs_decode)
print(f"Decode latency: {decode_latency:.6f} s, Peak memory: {decode_mem:.2f} MB")
