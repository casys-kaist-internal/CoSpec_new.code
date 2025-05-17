import json
import argparse
import time
import numpy as np
import asyncio
import aiohttp
from benchmark_dataset import (ShareGPTDataset)
from transformers import AutoTokenizer
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

@dataclass
class RequestMetrics:
    e2e_latency: float
    output_len: int
    prompt: str
    response: str

async def generate_text(session, prompt, server_url="http://localhost:8000", tokenizer=None):
    """
    Send a prompt to the FastAPI server and get the generated response asynchronously.
    
    Args:
        session (aiohttp.ClientSession): The aiohttp session to use
        prompt (str): The input prompt to generate text from
        server_url (str): The base URL of the FastAPI server
        tokenizer: The tokenizer to use for counting tokens
        
    Returns:
        dict: The response containing the prompt and generated text
    """
    # Endpoint for text generation
    endpoint = f"{server_url}/generate/"
    
    # Prepare the request payload
    payload = {
        "prompt": prompt
    }
    
    try:
        start_time = time.time()
        # Send POST request to the server
        async with session.post(endpoint, json=payload) as response:
            # Check if the request was successful
            response.raise_for_status()
            
            # Get the response
            result = await response.json()
            end_time = time.time()
            
            # Calculate metrics
            e2e_latency = end_time - start_time
            # Count tokens using the tokenizer
            output_len = len(tokenizer(result["response"], add_special_tokens=False).input_ids)
            
            return RequestMetrics(
                e2e_latency=e2e_latency,
                output_len=output_len,
                prompt=result["prompt"],
                response=result["response"]
            )
    
    except aiohttp.ClientError as e:
        print(f"Error occurred while making the request: {e}")
        return None

async def process_prompt(session, prompt, server_url, tokenizer):
    """Process a single prompt without waiting for delay."""
    result = await generate_text(session, prompt, server_url=server_url, tokenizer=tokenizer)
    if result:
        print("\nPrompt:", result.prompt)
        print("\nGenerated Response:", result.response)
    return result

def calculate_metrics(metrics: List[RequestMetrics]) -> dict:
    """Calculate benchmark metrics from request metrics."""
    if not metrics:
        return {
            "mean_latency_per_token": 0,
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency": 0
        }
    
    total_latency = sum(m.e2e_latency for m in metrics)
    total_tokens = sum(m.output_len for m in metrics)
    
    return {
        "mean_latency_per_token": total_latency / total_tokens if total_tokens > 0 else 0,
        "total_requests": len(metrics),
        "total_tokens": total_tokens,
        "total_latency": total_latency
    }

async def main_async():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Client for text generation with Poisson distributed request rates')
    parser.add_argument('--server-url', type=str, default="http://localhost:8000",
                      help='Base URL of the FastAPI server')
    parser.add_argument('--request-rate', type=float, default=1.0,
                      help='Average requests per second (lambda for Poisson distribution)')
    parser.add_argument('--dataset-path', type=str, default="ShareGPT_V3_unfiltered_cleaned_split.json",
                      help='Path to the dataset file')
    parser.add_argument('--random-seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--duration-minutes', type=float, default=5,
                      help='Duration to run the benchmark in minutes. If None, runs until dataset is exhausted.')
    args = parser.parse_args()

    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    sharegpt_dataset = ShareGPTDataset(random_seed=args.random_seed,
                      dataset_path="ShareGPT_V3_unfiltered_cleaned_split.json").sample_specinfer(tokenizer=tokenizer, num_requests=100)
    
    # Calculate end time if duration is specified
    end_time = None
    if args.duration_minutes is not None:
        end_time = time.time() + (args.duration_minutes * 60)
    
    # Create progress bar
    pbar = tqdm(total=len(sharegpt_dataset), unit='prompts', desc='Processing prompts')
    
    # Create aiohttp session
    async with aiohttp.ClientSession() as session:
        # Process prompts with Poisson distributed delays
        i = 0
        tasks = set()
        completed_metrics = []
        
        while True:
            # Check if we've reached the time limit
            if end_time is not None and time.time() >= end_time:
                pbar.close()
                print(f"\nReached time limit of {args.duration_minutes} minutes")
                break
                
            # Check if we've exhausted the dataset
            if i >= len(sharegpt_dataset):
                pbar.close()
                print("\nExhausted all prompts in dataset")
                break
            
            # Create and start new task
            prompt = sharegpt_dataset[i].prompt
            print("prompt length: ", len(prompt))
            task = asyncio.create_task(process_prompt(session, prompt, args.server_url, tokenizer))
            tasks.add(task)
            
            # Update progress bar to show sent requests
            pbar.set_postfix({"sent": i + 1, "completed": pbar.n})
            
            # Add callback to collect metrics when task completes
            def collect_metrics(task, metrics_list=completed_metrics, progress_bar=pbar):
                try:
                    result = task.result()
                    if result:
                        metrics_list.append(result)
                        progress_bar.update(1)  # Only update progress when request completes
                        progress_bar.set_postfix({"sent": i + 1, "completed": progress_bar.n})
                except Exception as e:
                    print(f"Error in task: {e}")
                finally:
                    tasks.discard(task)
            
            task.add_done_callback(collect_metrics)
            
            i += 1
            
            # Generate Poisson distributed delay for next request
            delay = np.random.exponential(1.0 / args.request_rate)
            await asyncio.sleep(delay)
        
        # Calculate and print metrics
        metrics = calculate_metrics(completed_metrics)
        print("\nBenchmark Results:")
        print(f"Mean latency per token: {metrics['mean_latency_per_token']:.4f} seconds")
        print(f"Total requests completed: {metrics['total_requests']}")
        print(f"Total tokens generated: {metrics['total_tokens']}")
        print(f"Total latency: {metrics['total_latency']:.2f} seconds")

        # wait for all tasks to complete
        await asyncio.gather(*tasks)

def main():
    asyncio.run(main_async())

if __name__ == "__main__":
    main() 