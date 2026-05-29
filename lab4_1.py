import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def main():
	# Model and tokenizer
	model_name = "Qwen/Qwen3-4B-Thinking-2507"
	print("Loading model and tokenizer...")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
	model.eval()

	# Prepare input
	context_len = 1000
	num_new_tokens = 3
	input_text = ("Once upon a time, " * ((context_len // 4) + 1))[:context_len]
	inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=context_len)
	inputs = {k: v.cuda() for k, v in inputs.items()}

	# Reset memory stats before timing
	torch.cuda.reset_peak_memory_stats()
	start = time.time()

	# Inference
	with torch.no_grad():
		output = model.generate(inputs["input_ids"], max_new_tokens=num_new_tokens)

	end = time.time()
	inference_time = end - start
	peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

	print(f"FP16 Inference time: {inference_time:.3f} seconds")
	print(f"FP16 Peak GPU memory usage: {peak_memory:.2f} MB")
	print("\nMemory measurement: \n- Model loaded and moved to GPU before measuring.\n- torch.cuda.reset_peak_memory_stats() called immediately before timed inference.\n- torch.cuda.max_memory_allocated() called after inference to get peak memory used during the run.\n- This ensures the measurement window is consistent and does not include temporary allocations from model loading.")

if __name__ == "__main__":
	main()
