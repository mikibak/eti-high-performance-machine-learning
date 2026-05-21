# ============================================================
# Task 3: CUDA Graphs + Max-Autotune (FIXED VERSION)
# ============================================================

import time
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

# ============================================================
# Configuration
# ============================================================

model_name = "Qwen/Qwen3-4B-Thinking-2507"

torch.set_float32_matmul_precision("high")

# Avoid silent compile fallback failures
torch._dynamo.config.suppress_errors = True

# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for this task."
    )

print("GPU:", torch.cuda.get_device_name(0))

# ============================================================
# Load model
# ============================================================

print("\nLoading model...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,

    # FIXED:
    # torch_dtype is deprecated
    dtype=torch.float16,
)

model = model.to(device)
model.eval()

# ============================================================
# Inputs
# ============================================================

# ------------------------------------------------------------
# PREFILL workload (long context)
# ------------------------------------------------------------

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

# ------------------------------------------------------------
# DECODE workload (very short context)
# ------------------------------------------------------------

prompt_decode = "Hello!"

inputs_decode = tokenizer(
    prompt_decode,
    return_tensors="pt",
)

inputs_decode = {
    k: v.to(device)
    for k, v in inputs_decode.items()
}

# ============================================================
# Baseline eager
# ============================================================

@torch.no_grad()
def eager_forward(**inputs):
    return model(**inputs)

# ============================================================
# Compiled variants
# ============================================================

print("\nCompiling models...")

# ------------------------------------------------------------
# 1. Standard compile
# ------------------------------------------------------------

compiled_standard = torch.compile(
    model.forward,
    mode="reduce-overhead",
    fullgraph=False,
)

# ------------------------------------------------------------
# 2. CUDA graphs enabled
# IMPORTANT:
# Cannot combine mode + options
# ------------------------------------------------------------

compiled_cudagraphs = torch.compile(
    model.forward,

    fullgraph=False,

    options={
        "triton.cudagraphs": True,
    },
)

# ------------------------------------------------------------
# 3. Max autotune
# ------------------------------------------------------------

compiled_autotune = torch.compile(
    model.forward,
    mode="max-autotune",
    fullgraph=False,
)

# ------------------------------------------------------------
# 4. CUDA graphs + max autotune
# IMPORTANT:
# Cannot combine mode + options
# ------------------------------------------------------------

compiled_both = torch.compile(
    model.forward,

    fullgraph=False,

    options={
        "triton.cudagraphs": True,
        "max_autotune": True,
    },
)

print("Compilation complete.\n")

# ============================================================
# Wrappers
# ============================================================

@torch.no_grad()
def forward_standard(**inputs):
    return compiled_standard(**inputs)

@torch.no_grad()
def forward_cudagraphs(**inputs):
    return compiled_cudagraphs(**inputs)

@torch.no_grad()
def forward_autotune(**inputs):
    return compiled_autotune(**inputs)

@torch.no_grad()
def forward_both(**inputs):
    return compiled_both(**inputs)

# ============================================================
# Warm-up
# ============================================================

print("Running warm-up...\n")

forward_fns = [
    eager_forward,
    forward_standard,
    forward_cudagraphs,
    forward_autotune,
    forward_both,
]

for fn in forward_fns:

    for _ in range(3):

        fn(**inputs_prefill)
        fn(**inputs_decode)

torch.cuda.synchronize()

print("Warm-up complete.\n")

# ============================================================
# Latency helper
# ============================================================

def measure_latency(
    forward_fn,
    inputs,
    n_iter=20,
):

    torch.cuda.synchronize()

    times = []

    for _ in range(n_iter):

        start = time.time()

        forward_fn(**inputs)

        torch.cuda.synchronize()

        end = time.time()

        times.append(end - start)

    return sum(times) / len(times)

# ============================================================
# Benchmark helper
# ============================================================

def benchmark(
    name,
    forward_fn,
):

    print(f"Running benchmark: {name}")

    prefill_latency = measure_latency(
        forward_fn,
        inputs_prefill,
    )

    decode_latency = measure_latency(
        forward_fn,
        inputs_decode,
    )

    print(
        f"Prefill latency: "
        f"{prefill_latency:.6f} s"
    )

    print(
        f"Decode latency : "
        f"{decode_latency:.6f} s"
    )

    print()

    return {
        "prefill": prefill_latency,
        "decode": decode_latency,
    }

# ============================================================
# Run benchmarks
# ============================================================

results = {}

results["eager"] = benchmark(
    "Eager",
    eager_forward,
)

results["compiled_standard"] = benchmark(
    "Compiled Standard",
    forward_standard,
)

results["compiled_cudagraphs"] = benchmark(
    "Compiled + CUDA Graphs",
    forward_cudagraphs,
)

results["compiled_autotune"] = benchmark(
    "Compiled + Max Autotune",
    forward_autotune,
)

results["compiled_both"] = benchmark(
    "Compiled + CUDA Graphs + Max Autotune",
    forward_both,
)

# ============================================================
# Summary table
# ============================================================

print("======================================================")
print("FINAL RESULTS")
print("======================================================")

header = (
    f"{'Configuration':35}"
    f"{'Prefill (s)':15}"
    f"{'Decode (s)':15}"
)

print(header)

for name, result in results.items():

    row = (
        f"{name:35}"
        f"{result['prefill']:<15.6f}"
        f"{result['decode']:<15.6f}"
    )

    print(row)

print("======================================================")

# ============================================================
# Speedups
# ============================================================

baseline_prefill = results["eager"]["prefill"]
baseline_decode = results["eager"]["decode"]

print("\n======================================================")
print("SPEEDUPS VS EAGER")
print("======================================================")

for name, result in results.items():

    if name == "eager":
        continue

    prefill_speedup = (
        baseline_prefill / result["prefill"]
    )

    decode_speedup = (
        baseline_decode / result["decode"]
    )

    print(f"{name}")

    print(
        f"  Prefill speedup: "
        f"{prefill_speedup:.3f}x"
    )

    print(
        f"  Decode speedup : "
        f"{decode_speedup:.3f}x"
    )

    print()

print("======================================================")
