import torch
import time
import os
import numpy as np
from typing import List, Dict, Tuple
import re
import csv
from vllm.logger import init_logger
from vllm.config import VllmConfig
logger = init_logger(__name__)

class CospecProfiler:
    def __init__(self, vllm_config: VllmConfig):
        self.profiling = False
        self.start_times = {}
        self.profile_results = {}
        self.current_key = None

        # Get GPU name and sanitize
        self.gpu_name = self._sanitize_filename(torch.cuda.get_device_name(0))
        self.target_model = self._sanitize_filename(vllm_config.model_config.model)
        self.draft_model = self._sanitize_filename(vllm_config.speculative_config.model)
        
        # Generate unique profile filename
        self.profile_file = (
            f"profile/{self.gpu_name}_"
            f"{self.target_model}_"
            f"{self.draft_model}_results.csv"
        )
        logger.info(f"Profile file: {self.profile_file}")

    def _sanitize_filename(self, model_name: str) -> str:
        """Convert model name to filesystem-safe identifier"""
        return re.sub(r"[^a-zA-Z0-9_-]", "_", model_name)[:64]

    def start(self):
        logger.info("Starting cospec profiler")
        self.profiling = True

    def stop(self):
        logger.info("Stopping cospec profiler")
        self.profiling = False

        try:
            os.makedirs(os.path.dirname(self.profile_file), exist_ok=True)
            
            with open(self.profile_file, "a") as f:  # Changed to append mode
                if os.stat(self.profile_file).st_size == 0:
                    f.write("name,running_queue_size,total_seq_len,num_lookahead_slots,duration\n")
                
                if not self.profile_results:
                    logger.warning("No profile results to write")
                    return
                    
                for key, durations in self.profile_results.items():
                    if not durations:
                        continue
                    key_str = ",".join(map(str, key))
                    for duration in durations:
                        f.write(f"{key_str},{duration}\n")
            
            logger.info(f"Successfully wrote {len(self.profile_results)} profile entries to {self.profile_file}")

        except Exception as e:
            logger.error(f"Failed to write profile results: {str(e)}")

    def maybe_load_cached_results(self) -> bool:
        if not os.path.exists(self.profile_file):
            logger.warning(f"Profile file {self.profile_file} does not exist")
            return False
        
        with open(self.profile_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) != 5:
                    logger.warning(f"Skipping invalid row: {row}")
                    continue
                
                try:
                    # Parse row: name,q,seq_len,lookahead,duration
                    key = (row[0], int(row[1]), int(row[2]), int(row[3]))
                    duration = float(row[4])
                    
                    if key not in self.profile_results:
                        self.profile_results[key] = []
                    self.profile_results[key].append(duration)
                    
                except (ValueError, IndexError) as e:
                    logger.warning(f"Skipping malformed row {row}: {str(e)}")

        logger.info(f"Loaded cached profile results from {self.profile_file}. Skipping profiling.")
        return True

    def start_marker(self, name:str):
        if not self.profiling:
            return 
        
        torch.cuda.synchronize()

        if name.startswith("run_speculative_decoding_step"):
            # parse name with . as delimiter
            running_queue_size, total_seq_len, num_lookahead_slots = name.split(".")[1:]
            self.current_key = (running_queue_size, total_seq_len, num_lookahead_slots)
            key = ("step", running_queue_size, total_seq_len, num_lookahead_slots)
        else:
            assert self.current_key is not None
            key = self.current_key
            if name == "draft":
                key = ("draft", key[0], key[1], key[2])
            elif name == "target":
                key = ("target", key[0], key[1], key[2])
            else:
                raise ValueError(f"Unknown name: {name}")

        self.start_times[key] = time.perf_counter()

    def stop_marker(self, name:str):
        if not self.profiling:
            return 
        
        torch.cuda.synchronize()
        assert self.current_key is not None
        key = self.current_key

        if name.startswith("run_speculative_decoding_step"):
            key = ("step", key[0], key[1], key[2])
        else:
            assert name == "draft" or name == "target"
            key = (name, key[0], key[1], key[2])

        assert key in self.start_times
        duration = time.perf_counter() - self.start_times[key]
        del self.start_times[key]

        if key not in self.profile_results:
            self.profile_results[key] = []
        self.profile_results[key].append(duration)

    def analyze(self):
        """Perform latency analysis and generate interactive report"""
        if not self.profile_results:
            logger.warning("No profile data to analyze")
            return

        processed_sets = self._process_profile_data()
        if not processed_sets:
            logger.warning("No complete sets for analysis")
            return

        self._generate_regression_models(processed_sets)

    def _process_profile_data(self) -> List[Dict]:
        """Organize raw profile data into structured sets"""
        param_keys = {
            (key[1], key[2], key[3])
            for key in self.profile_results
            if key[0] in ('draft', 'target', 'step')
        }

        processed_sets = []
        for q, seq_len, lookahead in param_keys:
            try:
                draft_dur = np.median(self.profile_results[('draft', q, seq_len, lookahead)])
                target_dur = np.median(self.profile_results[('target', q, seq_len, lookahead)])
                step_dur = np.median(self.profile_results[('step', q, seq_len, lookahead)])
                
                processed_sets.append({
                    'q': int(q),
                    'seq_len': int(seq_len),
                    'lookahead': int(lookahead),
                    'draft': draft_dur,
                    'target': target_dur,
                    'step': step_dur
                })
            except KeyError:
                continue
        return processed_sets

    def _generate_regression_models(self, data: List[Dict]):
        """Train and store regression models using optimized approach"""
        model_configs = [
            {
                'name': 'draft',
                'features': lambda d: [d['q'], d['seq_len']],
                'target': lambda d: d['draft'] / d['lookahead'],
                'coeff_names': ['q_coeff', 'seq_len_coeff']
            },
            {
                'name': 'target',
                'features': lambda d: [d['q'] * (d['lookahead'] + 1), d['seq_len']],
                'target': lambda d: d['target'],
                'coeff_names': ['adjq_coeff', 'seq_len_coeff']
            },
            {
                'name': 'prepost',
                'features': lambda d: [d['q'], d['lookahead']],
                'target': lambda d: d['step'] - (d['draft'] + d['target']),
                'coeff_names': ['q_coeff', 'lookahead_coeff']
            }
        ]

        self.models.clear()
        for config in model_configs:
            X = [config['features'](d) for d in data]
            y = [config['target'](d) for d in data]
            
            intercept, *coeffs = self._train_linear_model(X, y)
            logger.info(f"{config['name'].title()} model: y = {intercept:.2e} + " +
                        " + ".join(f"{c:.2e}*{n}" for c, n in zip(coeffs, config['coeff_names'])))

            self.models[config['name']] = {
                'intercept': intercept,
                **{name: coeff for name, coeff in zip(config['coeff_names'], coeffs)}
            }

    def _train_linear_model(self, X: List[List], y: List[float]) -> Tuple[float, ...]:
        """Efficient linear regression training with numpy"""
        X = np.column_stack([np.ones(len(X)), X])
        return np.linalg.lstsq(X, y, rcond=None)[0]

    def predict_draft_latency(self, batch_size: int, num_seq_len: int) -> float:
        model = self.models.get("draft")
        return (model['intercept'] 
                + model['q_coeff'] * batch_size
                + model['seq_len_coeff'] * num_seq_len)
    
    def predict_target_latency(self, num_tokens: int, num_seq_len: int) -> float:
        # For target model, batch size is not same as num_tokens. We input num_tokens to the model.
        model = self.models.get("target")
        return (model['intercept'] 
                + model['adjq_coeff'] * num_tokens
                + model['seq_len_coeff'] * num_seq_len)
        
    def predict_prepost_latency(self, batch_size: int, num_lookahead: int) -> float:
        model = self.models.get("prepost")
        return (model['intercept']
                + model['q_coeff'] * batch_size
                + model['lookahead_coeff'] * num_lookahead)
