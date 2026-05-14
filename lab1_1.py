# Model name
model_name = "Qwen/Qwen3-4B-Thinking-2507"

# Huggingface model card info
print("Qwen3-4B-Thinking-2507 has the following features:")
print("  Type: Causal Language Models")
print("  Training Stage: Pretraining & Post-training")
print("  Number of Parameters: 4.0B")
print("  Number of Parameters (Non-Embedding): 3.6B")
print("  Number of Layers: 36")
print("  Number of Attention Heads (GQA): 32 for Q and 8 for KV")
print("  Context Length: 262,144 natively.")
print()

vocab_size = 151936
hidden_size = 2560
num_layers = 36
num_heads = 32
intermediate_size = 6912

print("from config.json:\n")

print('"hidden_size": 2560')
print('"vocab_size": 151936')
print('"intermediate_size": 6912')
print('"num_hidden_layers": 36')
print('"num_attention_heads": 32')
print('"num_key_value_heads": 8\n')

embedding_params = vocab_size * hidden_size
print(f"Embedding parameters: {embedding_params:,}")

print(f"P = 12 * n_layers * d_model^2")
print(f"d_model^2 = {hidden_size}^2 = {hidden_size ** 2:,}")
print(f"12 * n_layers * d_model^2 = 12 * {num_layers} * {hidden_size ** 2:,} = {12 * num_layers * (hidden_size ** 2):,}")
total_params_formula = 12 * num_layers * (hidden_size ** 2)
print(f"\nP = {total_params_formula:,}")