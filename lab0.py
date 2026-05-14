from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Thinking-2507")
model = AutoModelForCausalLM.from_pretrained(
	"Qwen/Qwen3-4B-Thinking-2507",
	device_map="cpu"
)

messages = [
    {"role": "user", "content": "Give me a short introduction to large language model."},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

input_len = inputs.input_ids.shape[1]
print("input tokens: " + str(input_len))

outputs = model.generate(**inputs, 
    max_new_tokens=64,
    min_new_tokens=64,
    do_sample=False,
    num_beams=1,
    use_cache=True,
)
print("\n\n")
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
