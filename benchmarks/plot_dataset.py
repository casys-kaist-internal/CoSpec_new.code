from benchmark_dataset import (AIMODataset, ASRDataset, BurstGPTDataset,
                               ConversationDataset, HuggingFaceDataset,
                               InstructCoderDataset, RandomDataset,
                                SampleRequest, ShareGPTDataset, SonnetDataset,
                                VisionArenaDataset, GSM8KDataset, NaturalQuestionsDataset,
                                HumanEvalDataset, PythonAlpacaDataset, OrcaMathDataset)
try:
    from vllm.transformers_utils.tokenizer import get_tokenizer
except ImportError:
    from backend_request_func import get_tokenizer

import matplotlib.pyplot as plt
import numpy as np

seed = 42

tokenizer_id = "facebook/opt-6.7b"
tokenizer_mode = "auto"

tokenizer = get_tokenizer(tokenizer_id,
                        tokenizer_mode=tokenizer_mode,
                        trust_remote_code=True)

sharegpt_dataset = ShareGPTDataset(random_seed=seed,
                    dataset_path="ShareGPT_V3_unfiltered_cleaned_split.json").sample_all(tokenizer=tokenizer)
humaneval_dataset = HumanEvalDataset(random_seed=seed,
                    dataset_path="openai/openai_humaneval", 
                    dataset_split="test").sample_all(tokenizer=tokenizer)
python_alpaca_dataset = PythonAlpacaDataset(random_seed=seed,
                    dataset_path="Vezora/Tested-143k-Python-Alpaca", 
                    dataset_split="train").sample_all(tokenizer=tokenizer)
orca_math_dataset = OrcaMathDataset(random_seed=seed,
                    dataset_path="microsoft/orca-math-word-problems-200k", 
                    dataset_split="train").sample_all(tokenizer=tokenizer)

# Extract input and output lengths
datasets = {
    'ShareGPT': sharegpt_dataset,
    'HumanEval': humaneval_dataset,
    'PythonAlpaca': python_alpaca_dataset,
    'OrcaMath': orca_math_dataset
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



