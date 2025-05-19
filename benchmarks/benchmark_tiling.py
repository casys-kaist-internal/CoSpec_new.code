from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import numpy as np
import pandas as pd
import os

# Load model and tokenizer
model_name = "facebook/opt-6.7b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# # Print model configuration
# print("\nModel Configuration:")
# print(f"Model type: {model.config.model_type}")
# print(f"Hidden size: {model.config.hidden_size}")
# print(f"Number of layers: {model.config.num_hidden_layers}")
# print(f"Number of attention heads: {model.config.num_attention_heads}")

# # Analyze linear layers
# print("\nLinear Layer Shapes:")
# for name, module in model.named_modules():
#     if isinstance(module, torch.nn.Linear):
#         print(f"{name}:")
#         print(f"  Input shape: {module.in_features}")
#         print(f"  Output shape: {module.out_features}")
#         print(f"  Weight shape: {module.weight.shape}")
#         print(f"  Bias shape: {module.bias.shape if module.bias is not None else None}")

# Benchmark function
def benchmark_inference(input_length):
    # Create input tokens
    input_text = "Hello" * (input_length // 5 + 1)  # Ensure we have enough tokens
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=input_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(**inputs)
    
    # Measure latency
    latencies = []
    with torch.no_grad():
        for _ in range(10):  # Run 10 times for each length
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = model(**inputs)
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds
            latencies.append(duration)
    
    return np.mean(latencies), np.std(latencies)

# Run benchmarks and collect results
print("\nRunning benchmarks for different input lengths...")
results = []

for length in range(1, 257):
    avg_latency, std_latency = benchmark_inference(length)
    results.append({
        'input_length': length,
        'avg_duration_ms': avg_latency,
        'std_duration_ms': std_latency
    })
    print(f"Input Length: {length:3d} | Average Duration: {avg_latency:8.2f} ms | Std Dev: {std_latency:8.2f} ms")

# Create DataFrame and save to CSV
df = pd.DataFrame(results)
output_dir = "benchmark_results"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "tiling_benchmark_results.csv")
df.to_csv(output_file, index=False)
print(f"\nResults saved to {output_file}")

# Print summary statistics
print("\nSummary Statistics:")
print(df.describe())
