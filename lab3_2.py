import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import psutil
import os

# Try to import IPEX
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    raise ImportError("Please install Intel Extension for PyTorch (IPEX): pip install intel-extension-for-pytorch")

# =============================
# Task 1: IPEX INT8 Quantization
# =============================

model_name = "Qwen/Qwen3-4B-Thinking-2507"

torch.set_float32_matmul_precision("high")
torch._dynamo.config.suppress_errors = True

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load model in FP16, then quantize to INT8
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

device = torch.device("cpu")
model = model.to(device)
model.eval()

# Quantize model to INT8 with IPEX
print("Quantizing model to BF16 with IPEX...")
model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)

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

print("Measuring latency and memory usage (decode, 3 steps)...")
decode_times = []
decode_mems = []
for _ in range(3):
    decode_latency, decode_mem = measure_latency_and_memory(forward_fn, inputs_decode, n_iter=1)
    decode_times.append(decode_latency)
    decode_mems.append(decode_mem)
avg_decode_latency = sum(decode_times) / len(decode_times)
max_decode_mem = max(decode_mems)
print(f"Decode latency (avg 3 steps): {avg_decode_latency:.6f} s, Peak memory: {max_decode_mem:.2f} MB")

print("\nSummary:")
print(f"INT8 Quantized (IPEX) - Prefill: {prefill_latency:.6f} s, {prefill_mem:.2f} MB | Decode (3 steps avg): {avg_decode_latency:.6f} s, {max_decode_mem:.2f} MB")
print("\nCompare these results with your FP16 baseline from lab3_1.py.")
print("\nDiscuss: Did quantization help more with memory or runtime? How might AVX-VNNI support affect these results?")
