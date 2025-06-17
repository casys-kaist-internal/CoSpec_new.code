# Benchmark Experiments

## Main Experiments

### Experiment A
- **Target Model**: OPT-6.7B
- **Draft Model**: OPT-125M
- **Dataset**: Math500
- **Request Rates**: 2, 4, 6, 8, 10, 12, 14 req/s
- **GPU**: A6000
- **Target TP Size**: 1
- **Draft TP Size**: 1

### Experiment B
- **Target Model**: Llama-13B
- **Draft Model**: Llama-68M
- **Dataset**: Alpaca
- **Request Rates**: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20 req/s
- **GPU**: A6000
- **Target TP Size**: 2
- **Draft TP Size**: 1

### Experiment C
- **Target Model**: OPT-13B
- **Draft Model**: OPT-125M
- **Dataset**: ShareGPT
- **Request Rates**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 req/s
- **GPU**: A100
- **Target TP Size**: 1
- **Draft TP Size**: 1

### Experiment D
- **Target Model**: OPT-30B
- **Draft Model**: OPT-350M
- **Dataset**: OpenCode
- **Request Rates**: TBD
- **GPU**: A100
- **Target TP Size**: 2
- **Draft TP Size**: 2

### Experiment E
- **Target Model**: OPT-30B
- **Draft Model**: OPT-350M
- **Dataset**: OpenCode
- **Request Rates**: TBD
- **GPU**: H200
- **Target TP Size**: 1
- **Draft TP Size**: 1

## Individual Technique Experiments

### Experiment F
- **Target Model**: OPT-6.7B
- **Draft Model**: OPT-125M
- **Dataset**: Math500
- **Request Rates**: 10 req/s
- **GPU**: A6000
- **Target TP Size**: 1
- **Draft TP Size**: 1

### Experiment G
- **Target Model**: OPT-13B
- **Draft Model**: OPT-125M
- **Dataset**: ShareGPT
- **Request Rates**: 9 req/s
- **GPU**: A100
- **Target TP Size**: 1
- **Draft TP Size**: 1

### Experiment H
- **Target Model**: OPT-30B
- **Draft Model**: OPT-350M
- **Dataset**: OpenCode
- **Request Rates**: 5 req/s
- **GPU**: H200
- **Target TP Size**: 1
- **Draft TP Size**: 1

## Selective Validation Experiments

### Experiment I
- **Target Model**: OPT-6.7B
- **Draft Model**: 125M
- **Dataset**: Math500
- **Request Rates**: 10 req/s
- **GPU**: A6000
- **Target TP Size**: 1
- **Draft TP Size**: 1

### Experiment J
- **Target Model**: OPT-13B
- **Draft Model**: 125M
- **Dataset**: ShareGPT
- **Request Rates**: 9 req/s
- **GPU**: A100
- **Target TP Size**: 1
- **Draft TP Size**: 1

### Experiment K
- **Target Model**: Qwen3-32B
- **Draft Model**: Qwen3-0.6B
- **Dataset**: Open-Platypus
- **Model Path**: Qwen/Qwen3-32B
- **Request Rates**: TBD
- **GPU**: H200
- **Target TP Size**: 1
- **Draft TP Size**: 1
