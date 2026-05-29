import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from modelopt.torch.quantization import quantize

import modelopt.torch.quantization as mtq

def get_quant_config(recipe_name):
	if recipe_name == "fp8":
		return mtq.FP8_DEFAULT_CFG
	elif recipe_name == "int8-smoothquant":
		return mtq.INT8_SMOOTHQUANT_CFG
	elif recipe_name == "int4-awq":
		return mtq.INT4_AWQ_CFG
	raise ValueError(f"Unknown recipe: {recipe_name}")

def measure_inference(model, tokenizer, context_len=1000, num_new_tokens=3):
	input_text = ("Once upon a time, " * ((context_len // 4) + 1))[:context_len]
	inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=context_len)
	inputs = {k: v.cuda() for k, v in inputs.items()}
	torch.cuda.reset_peak_memory_stats()
	start = time.time()
	with torch.no_grad():
		_ = model.generate(inputs["input_ids"], max_new_tokens=num_new_tokens)
	end = time.time()
	inference_time = end - start
	peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
	return inference_time, peak_memory

def run_quantization_experiment(model_name, recipe_name, context_len=1000, num_new_tokens=3):
	print(f"\n--- {recipe_name} quantization ---")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
	model.eval()
	# Quantize
	quant_config = get_quant_config(recipe_name)
	
	def calibrate_loop(m):
		input_text = "Once upon a time, there was a quantization experiment."
		inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
		with torch.no_grad():
			m(**inputs)

	quantized_model = quantize(model, config=quant_config, forward_loop=calibrate_loop)
	quantized_model.eval()
	del model
	torch.cuda.empty_cache()
	# Measure
	inference_time, peak_memory = measure_inference(quantized_model, tokenizer, context_len, num_new_tokens)
	print(f"{recipe_name} Inference time: {inference_time:.3f} seconds")
	print(f"{recipe_name} Peak GPU memory usage: {peak_memory:.2f} MB")
	del quantized_model
	torch.cuda.empty_cache()
	return inference_time, peak_memory

def main():
	model_name = "Qwen/Qwen3-4B-Thinking-2507"
	context_len = 1000
	num_new_tokens = 3

	# FP16 baseline (for comparison)
	print("\n--- FP16 baseline ---")
	tokenizer = AutoTokenizer.from_pretrained(model_name)
	model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
	model.eval()
	fp16_time, fp16_mem = measure_inference(model, tokenizer, context_len, num_new_tokens)
	print(f"FP16 Inference time: {fp16_time:.3f} seconds")
	print(f"FP16 Peak GPU memory usage: {fp16_mem:.2f} MB")
	del model
	torch.cuda.empty_cache()

	# Quantization recipes
	results = {}
	for recipe_name in ["fp8", "int8-smoothquant", "int4-awq"]:
		q_time, q_mem = run_quantization_experiment(model_name, recipe_name, context_len, num_new_tokens)
		results[recipe_name] = {"time": q_time, "mem": q_mem}

	# Summary
	print("\nSummary:")
	print(f"{'Recipe':<18}{'Latency (s)':<15}{'Peak Memory (MB)':<15}")
	print(f"{'FP16':<18}{fp16_time:<15.3f}{fp16_mem:<15.2f}")
	for k, v in results.items():
		print(f"{k:<18}{v['time']:<15.3f}{v['mem']:<15.2f}")

	# Best memory and latency
	best_mem = min(results.items(), key=lambda x: x[1]['mem'])
	best_time = min(results.items(), key=lambda x: x[1]['time'])
	print(f"\nBest memory reduction: {best_mem[0]} ({best_mem[1]['mem']:.2f} MB)")
	print(f"Best end-to-end latency: {best_time[0]} ({best_time[1]['time']:.3f} s)")

if __name__ == "__main__":
	main()
