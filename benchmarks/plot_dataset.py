from benchmark_dataset import (AIMODataset, ASRDataset, BurstGPTDataset,
                               ConversationDataset, HuggingFaceDataset,
                               InstructCoderDataset, RandomDataset,
                                SampleRequest, ShareGPTDataset, SonnetDataset,
                                VisionArenaDataset, GSM8KDataset, NaturalQuestionsDataset,
                                HumanEvalDataset, PythonAlpacaDataset, OpenCodeInstructDataset,
                                OpenMathInstructDataset)
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

import matplotlib.pyplot as plt
import numpy as np
import json
seed = 42

tokenizer_id = "facebook/opt-6.7b"
tokenizer_mode = "auto"

tokenizer = get_tokenizer(tokenizer_id,
                        tokenizer_mode=tokenizer_mode,
                        trust_remote_code=True)

sharegpt_dataset = ShareGPTDataset(random_seed=seed,
                    dataset_path="ShareGPT_V3_unfiltered_cleaned_split.json").sample_all(tokenizer=tokenizer)
# humaneval_dataset = HumanEvalDataset(random_seed=seed,
#                     dataset_path="openai/openai_humaneval", 
#                     dataset_split="test").sample_all(tokenizer=tokenizer)
# gsm8k_dataset = GSM8KDataset(random_seed=seed,
#                     dataset_path="openai/gsm8k", 
#                     dataset_subset="main",
#                     dataset_split="train").sample_all(tokenizer=tokenizer)
openmath_dataset = OpenMathInstructDataset(random_seed=seed,
                    dataset_path="nvidia/OpenMathInstruct-2",
                    dataset_split="train").sample_all(tokenizer=tokenizer)
# python_alpaca_dataset = PythonAlpacaDataset(random_seed=seed,
#                     dataset_path="Vezora/Tested-143k-Python-Alpaca", 
#                     dataset_split="train").sample_all(tokenizer=tokenizer)
open_code_instruct_dataset = OpenCodeInstructDataset(random_seed=seed,
                    dataset_path="nvidia/OpenCodeInstruct", 
                    dataset_split="train").sample_all(tokenizer=tokenizer)
# open_math_reasoning_dataset = OpenMathReasoningDataset(random_seed=seed,
#                     dataset_path="nvidia/OpenMathReasoning", 
#                     dataset_split="cot").sample_all(tokenizer=tokenizer)

# # Extract input and output lengths
# datasets = {
#     'ShareGPT': sharegpt_dataset,
#     'PythonAlpaca': python_alpaca_dataset,
#     'HumanEval': humaneval_dataset,
#     'OpenCodeInstruct': open_code_instruct_dataset,
#     'OpenMathReasoning': open_math_reasoning_dataset
# }

# Save prompts to JSON file
# with open("sharegpt_prompts.json", "w") as f:
#     json.dump([sample.prompt for sample in sharegpt_dataset], f)

datasets = {
    'ShareGPT': sharegpt_dataset,
    'OpenMathInstruct': openmath_dataset,
    'OpenCodeInstruct': open_code_instruct_dataset,
}

# Create figure with subplots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Dataset Length Distributions', fontsize=16)

# Plot input lengths
for name, dataset in datasets.items():
    input_lengths = [req.prompt_len for req in dataset]
    axes[0].hist(input_lengths, bins=30, alpha=0.5, label=name, density=True)
axes[0].set_title('Input Length Distribution')
axes[0].set_xlabel('Number of Tokens')
axes[0].set_ylabel('Density')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot output lengths
for name, dataset in datasets.items():
    output_lengths = [req.expected_output_len for req in dataset]
    axes[1].hist(output_lengths, bins=30, alpha=0.5, label=name, density=True)
axes[1].set_title('Output Length Distribution')
axes[1].set_xlabel('Number of Tokens')
axes[1].set_ylabel('Density')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dataset_lengths.png', dpi=300, bbox_inches='tight')
plt.close()

# Print summary statistics
print("\nDataset Length Statistics:")
print("-" * 50)
for name, dataset in datasets.items():
    input_lengths = [req.prompt_len for req in dataset]
    output_lengths = [req.expected_output_len for req in dataset]
    print(f"\n{name}:")
    print(f"Input lengths - Mean: {np.mean(input_lengths):.1f}, Median: {np.median(input_lengths):.1f}, "
          f"Min: {min(input_lengths)}, Max: {max(input_lengths)}")
    print(f"Output lengths - Mean: {np.mean(output_lengths):.1f}, Median: {np.median(output_lengths):.1f}, "
          f"Min: {min(output_lengths)}, Max: {max(output_lengths)}")