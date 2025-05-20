import torch
import time
from transformers import AutoModelForCausalLM
import numpy as np
import csv
import os

def get_first_transformer_layer():
    # Load the model
    model = AutoModelForCausalLM.from_pretrained("facebook/opt-6.7b", torch_dtype=torch.float16)
    
    # Find the first transformer layer's linear layers
    transformer_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Look for layers in the first transformer block
            if 'layers.0' in name:
                in_features = module.in_features
                out_features = module.out_features
                print(f"Found linear layer in first transformer: {name}")
                print(f"Input features: {in_features}")
                print(f"Output features: {out_features}")
                transformer_layers.append((name, in_features, out_features))
    
    return transformer_layers

def remove_outliers(times):
    """Remove outliers using 2 standard deviations from mean"""
    times = np.array(times)
    mean = np.mean(times)
    std = np.std(times)
    mask = np.abs(times - mean) <= 2 * std
    return times[mask]

def profile_linear_layer(name, in_features, out_features, batch_sizes, num_runs=100):
    # Create the linear layer
    linear = torch.nn.Linear(in_features, out_features).cuda().half()
    
    # Create all input tensors at once
    inputs = {bs: torch.randn(bs, in_features, device='cuda', dtype=torch.float16) 
             for bs in batch_sizes}
    
    # Warm up
    x = torch.randn(1, in_features, device='cuda', dtype=torch.float16)
    for _ in range(10):
        _ = linear(x)
    torch.cuda.synchronize()
    
    results = []
    for batch_size in batch_sizes:
        x = inputs[batch_size]
        
        # Collect individual run times
        run_times = []
        for _ in range(num_runs):
            torch.cuda.synchronize()
            start_time = time.time()
            _ = linear(x)
            torch.cuda.synchronize()
            end_time = time.time()
            run_times.append(end_time - start_time)
        
        # Remove outliers and calculate mean
        filtered_times = remove_outliers(run_times)
        avg_time = np.mean(filtered_times)
        
        results.append((batch_size, avg_time))
        print(f"Layer: {name}, Batch size: {batch_size}")
        print(f"  Raw times: mean={np.mean(run_times)*1000:.6f}ms, std={np.std(run_times)*1000:.6f}ms")
        print(f"  After outlier removal: mean={avg_time*1000:.6f}ms, samples={len(filtered_times)}/{num_runs}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    return results

def save_results_to_csv(all_results, output_file="transformer_layer_profiling.csv"):
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow(['input_tokens', 'mean_latency'])
        
        # Get all batch sizes from the first layer's results
        batch_sizes = [r[0] for r in next(iter(all_results.values()))]
        
        # Write data for each batch size
        for i, batch_size in enumerate(batch_sizes):
            total_time = 0
            
            # Calculate total time across all layers
            for layer_name in all_results.keys():
                layer_time = all_results[layer_name][i][1] * 1000  # Convert to ms
                total_time += layer_time
            
            # Write only batch size and total time
            writer.writerow([batch_size, f"{total_time:.6f}"])
    
    print(f"\nResults saved to {output_file}")

def main():
    # Get the first transformer layer's linear layers
    transformer_layers = get_first_transformer_layer()
    if not transformer_layers:
        print("No transformer layers found in the model")
        return
    
    # Define batch sizes to test
    batch_sizes = range(8, 2049, 8)
    
    # Store results for each layer
    all_results = {}
    
    # Profile each linear layer in the first transformer
    for name, in_features, out_features in transformer_layers:
        print(f"\nProfiling layer: {name}")
        results = profile_linear_layer(name, in_features, out_features, batch_sizes)
        all_results[name] = results
        
        # Print results in a table format
        print(f"\nResults Summary for {name}:")
        print("Batch Size | Time (ms)")
        print("-" * 20)
        for batch_size, time_taken in results:
            print(f"{batch_size:10d} | {time_taken*1000:8.6f}")
    
    # Save results to CSV
    save_results_to_csv(all_results)

if __name__ == "__main__":
    main()
