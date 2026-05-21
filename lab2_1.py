import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen3-4B-Thinking-2507"

torch.set_float32_matmul_precision("high")

# Avoid silent compile failures
torch._dynamo.config.suppress_errors = True
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


compiled_forward_model = torch.compile(
    model.forward,
    mode="reduce-overhead",
    fullgraph=False,
)

@torch.no_grad()
def eager_forward(**inputs):
    return model(**inputs)

@torch.no_grad()
def compiled_forward(**inputs):
    return compiled_forward_model(**inputs)


print("Running warm-up...")

for _ in range(3):
    eager_forward(**inputs_prefill)
    eager_forward(**inputs_decode)

for _ in range(3):
    compiled_forward(**inputs_prefill)
    compiled_forward(**inputs_decode)

if torch.cuda.is_available():
    torch.cuda.synchronize()

print("Warm-up complete.\n")



def measure_latency(forward_fn, inputs, n_iter=10):

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []

    for _ in range(n_iter):

        start = time.time()

        forward_fn(**inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end = time.time()

        times.append(end - start)

    return sum(times) / len(times)


print("Measuring latency...\n")

prefill_eager = measure_latency(
    eager_forward,
    inputs_prefill,
)

prefill_compiled = measure_latency(
    compiled_forward,
    inputs_prefill,
)

decode_eager = measure_latency(
    eager_forward,
    inputs_decode,
)

decode_compiled = measure_latency(
    compiled_forward,
    inputs_decode,
)


print("========== LATENCY RESULTS ==========")

print(f"Prefill eager latency   : {prefill_eager:.6f} s")
print(f"Prefill compiled latency: {prefill_compiled:.6f} s")

print()

print(f"Decode eager latency    : {decode_eager:.6f} s")
print(f"Decode compiled latency : {decode_compiled:.6f} s")

print("=====================================\n")


def count_cuda_kernels(forward_fn, inputs):

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:

        forward_fn(**inputs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    cuda_events = []

    for evt in prof.key_averages():

        # Keep only CUDA ops
        if evt.device_type == torch.profiler.ProfilerActivity.CUDA:

            cuda_events.append((evt.key, evt.count))

    total_cuda_kernels = sum(count for _, count in cuda_events)

    return total_cuda_kernels, cuda_events


if torch.cuda.is_available():

    print("Counting CUDA kernels...\n")

    # Eager
    eager_prefill_kernels, eager_prefill_ops = count_cuda_kernels(
        eager_forward,
        inputs_prefill,
    )

    eager_decode_kernels, eager_decode_ops = count_cuda_kernels(
        eager_forward,
        inputs_decode,
    )

    # Compiled
    compiled_prefill_kernels, compiled_prefill_ops = count_cuda_kernels(
        compiled_forward,
        inputs_prefill,
    )

    compiled_decode_kernels, compiled_decode_ops = count_cuda_kernels(
        compiled_forward,
        inputs_decode,
    )

    print("========== KERNEL RESULTS ==========")

    print(f"Prefill eager kernels   : {eager_prefill_kernels}")
    print(f"Prefill compiled kernels: {compiled_prefill_kernels}")

    print()

    print(f"Decode eager kernels    : {eager_decode_kernels}")
    print(f"Decode compiled kernels : {compiled_decode_kernels}")

    print("====================================\n")


    print("Sample eager CUDA ops:")
    for name, count in eager_prefill_ops[:15]:
        print(f"{name:<40} count={count}")

    print()

    print("Sample compiled CUDA ops:")
    for name, count in compiled_prefill_ops[:15]:
        print(f"{name:<40} count={count}")

    print()

else:
    print("CUDA not available. Kernel counting skipped.")