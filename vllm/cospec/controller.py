from __future__ import annotations
import asyncio
from typing import Dict, Optional, Protocol, List
import numpy as np
from dataclasses import dataclass
from enum import Enum
from vllm.cospec.shared_memory import CoSpecSharedMemory

class LoadBalanceStrategy(Enum):
    ROUND_ROBIN = "round_robin"

@dataclass
class EngineState:
    engine_id: int
    active_requests: int = 0
    healthy: bool = True
    last_latency: float = 0.0

class CoSpecController:
    def __init__(
        self,
        engine1,
        engine2,
        strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN,
        shm_size: int = 2 * 1024**3
    ):
        self.engine1 = engine1
        self.engine2 = engine2
        self.shm = CoSpecSharedMemory(size=shm_size)
        self.strategy = self._init_strategy(strategy)
        self._lock = asyncio.Lock()
        self._strategy_lock = asyncio.Lock()
        
        # Shared memory buffer for real-time metrics (128 bytes per engine)
        self.metrics_buffer = self.shm.read_data(
            shape=(256,), 
            dtype=np.uint8,
            offset=0
        )

    def _init_strategy(self, strategy: LoadBalanceStrategy):
        strategies = {
            LoadBalanceStrategy.ROUND_ROBIN: RoundRobinStrategy(),
        }
        return strategies[strategy]

    def select_engine(self):
        engine_id = self.strategy.select_engine()
        if engine_id == 0:
            return self.engine1
        else:
            return self.engine2

class RoundRobinStrategy:
    def __init__(self):
        self._counter = 0
        
    def select_engine(self) -> int:
        self._counter = (self._counter + 1) % 2
        return self._counter