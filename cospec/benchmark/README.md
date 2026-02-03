# CoSpec Evaluation Framework

This directory contains the evaluation framework for CoSpec, designed to produce publication-quality results following OSDI/SOSP standards.

## Quick Start

```bash
# 1. Start MPS (required for CoSpec)
bash cospec/scripts/start_mps.sh

# 2. Download ShareGPT dataset (auto-downloaded if not present)
# Or manually download to project root

# 3. Run full evaluation (takes ~5 hours)
./run_evaluation.sh

# 4. Quick test (fewer configurations, ~1 hour)
./run_evaluation.sh --quick

# 5. Generate plots
python plot_results.py --results-dir results/evaluation_YYYYMMDD_HHMMSS
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSPEC` | `0` | Enable CoSpec (SM partitioning + MPS) |
| `COSPEC_TARGET_SM_RATIO` | `0.7` | Target model SM fraction (0.0-1.0) |
| `VLLM_USE_V1` | `1` | Must be `0` for speculative decoding |

## Evaluation Components

### 1. Benchmark Script (`bench_serving.py`)

Async HTTP benchmark client with:
- Poisson-distributed request arrivals
- Duration-based or count-based runs
- TTFT, TPOT, ITL, E2E latency metrics
- Support for ShareGPT, random, sonnet, burstgpt datasets

```bash
# Fixed number of prompts
python bench_serving.py --model meta-llama/Llama-3.1-8B \
    --dataset sharegpt --dataset-path ShareGPT_V3.json \
    --num-prompts 200 --request-rate 4.0

# Duration-based (recommended for evaluation)
python bench_serving.py --model meta-llama/Llama-3.1-8B \
    --dataset sharegpt --dataset-path ShareGPT_V3.json \
    --duration 300 --request-rate 4.0
```

### 2. Evaluation Orchestrator (`run_evaluation.sh`)

Runs the full evaluation matrix:
- **Experiment 1**: Main comparison (AR vs Vanilla SD vs CoSpec) across 10 request rates
- **Experiment 3**: SM ratio ablation (0.5, 0.6, 0.7, 0.8, 0.9)
- **Experiment 4**: Gamma (speculation length) ablation (3, 5, 7)

Options:
```bash
./run_evaluation.sh --quick            # Fewer rates, 1 repeat
./run_evaluation.sh --experiment 1     # Run specific experiment
./run_evaluation.sh --config cospec    # Run specific config only
./run_evaluation.sh --skip-warmup      # Skip warmup phase
./run_evaluation.sh --target-model MODEL --draft-model MODEL
```

### 3. Plotting Script (`plot_results.py`)

Generates publication-quality figures:
- **Figure 1**: Latency-throughput curves (P99 TTFT vs request rate)
- **Figure 3**: SM ratio ablation (throughput + latency vs SM ratio)
- **Figure 4**: Gamma sensitivity (bar chart)
- Summary tables in CSV and LaTeX

```bash
python plot_results.py --results-dir results/evaluation_XXXXXX
python plot_results.py --results-csv custom_results.csv --output-dir plots/
```

## Server Launch Commands

```bash
# AR Baseline (no speculation)
VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --port 8100 --gpu-memory-utilization 0.85

# Vanilla SD (sequential draft→verify)
VLLM_USE_V1=0 python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --speculative-model meta-llama/Llama-3.1-1B \
    --num-speculative-tokens 5 \
    --port 8100 --gpu-memory-utilization 0.85

# CoSpec (concurrent SM-partitioned)
COSPEC=1 COSPEC_TARGET_SM_RATIO=0.7 VLLM_USE_V1=0 \
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-8B \
    --speculative-model meta-llama/Llama-3.1-1B \
    --num-speculative-tokens 5 \
    --port 8100 --gpu-memory-utilization 0.85 --enforce-eager
```

## Output Format

Results are saved in CSV format with columns:
- `config`: ar, vanilla_sd, cospec, cospec_ablation, etc.
- `gamma`: speculation length (0 for AR)
- `sm_ratio`: target SM ratio (1.0 for non-CoSpec)
- `request_rate`: requests per second
- `repeat`: repetition number (for statistical significance)
- Metrics: completed, failed, throughput, TTFT/TPOT/ITL/E2E (mean, p50, p90, p99)

## Expected Results

Based on CoSpec paper claims:
- CoSpec achieves higher throughput than Vanilla SD at same latency
- Optimal SM ratio is typically 0.6-0.8 for 8B target / 1B draft
- CoSpec maintains similar acceptance rate to Vanilla SD

## Model Configurations

| Target | Draft | Size Ratio | Memory |
|--------|-------|------------|--------|
| Llama-3.1-8B | Llama-3.1-1B | 8:1 | ~20GB |
| Qwen3-8B | Qwen3-0.6B | ~13:1 | ~18GB |

## Troubleshooting

1. **MPS not running**: `bash cospec/scripts/start_mps.sh`
2. **V1 engine error**: Set `VLLM_USE_V1=0`
3. **OOM**: Reduce `--gpu-memory-utilization` or batch size
4. **Server startup timeout**: Check server logs in results directory
