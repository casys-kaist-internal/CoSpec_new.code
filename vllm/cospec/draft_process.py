# SPDX-License-Identifier: Apache-2.0
"""CoSpec Draft Process Management.

Handles spawning the draft model worker in a separate OS process for MPS
SM-partitioned concurrency, as well as cleanup at exit.
"""

import atexit
import multiprocessing as mp
import os
import socket
import subprocess
import traceback
import weakref
from typing import Any, Dict, List, Optional

from vllm.cospec import cleanup_cospec_resources
from vllm.logger import init_logger

logger = init_logger(__name__)

# Module-level registry for cleanup at exit
_cospec_instances: List[weakref.ref] = []
_atexit_registered = False


def assert_mps_running() -> None:
    """Assert that NVIDIA MPS daemon is running.

    CoSpec requires MPS for true concurrent execution of target and draft
    processes on the same GPU with SM partitioning.
    """
    # Check standard MPS pipe directory locations
    mps_pipe_dir = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
    if mps_pipe_dir and os.path.isdir(mps_pipe_dir):
        return

    # Check default location
    if os.path.isdir("/tmp/nvidia-mps"):
        return

    # Fall back to checking if MPS control daemon process is running
    try:
        result = subprocess.run(
            ["pgrep", "-f", "nvidia-cuda-mps-control"],
            capture_output=True, timeout=5)
        if result.returncode == 0:
            return
    except Exception:
        pass

    raise RuntimeError(
        "CoSpec requires NVIDIA MPS (Multi-Process Service) to be running. "
        "Start MPS with: cospec/scripts/start_mps.sh "
        "or: nvidia-cuda-mps-control -d")


def init_draft_process(sdw: Any) -> None:
    """Spawn a separate OS process for the draft model via MPS.

    Creates a pipe for RPC communication, spawns the child process,
    and sets up the orchestrator with always-colocated cost model.

    Args:
        sdw: The SpecDecodeWorker instance.
    """
    from vllm.config import envs
    from vllm.cospec.orchestrator import CoSpecOrchestrator
    from vllm.cospec.shared_memory import SharedLogitBuffer
    from vllm.cospec.worker_rpc import DraftWorkerRPC, create_draft_worker_pipe

    parent_conn, child_conn = create_draft_worker_pipe()

    # Create shared logit buffer (target=owner, draft=client).
    # Must be created before spawning draft so the IPC handle file exists.
    # Use PID-based instance_id to avoid collisions between multiple
    # CoSpec instances on the same machine.
    num_spec_tokens = sdw.max_spec_tokens
    max_batch = sdw.scorer_worker.vllm_config.scheduler_config.max_num_seqs
    vocab_size = sdw._vocab_size
    instance_id = str(os.getpid())
    shared_logit_buffer = SharedLogitBuffer(
        max_batch=max_batch,
        max_spec_tokens=num_spec_tokens,
        vocab_size=vocab_size,
        dtype=sdw.probs_dtype,
        mode="owner",
        instance_id=instance_id,
    )
    sdw._shared_logit_buffer = shared_logit_buffer

    # Spawn draft worker as a separate OS process for true MPS
    # SM-partitioned concurrency.
    logit_buffer_config = {
        "max_batch": max_batch,
        "max_spec_tokens": num_spec_tokens,
        "vocab_size": vocab_size,
        "dtype": sdw.probs_dtype,
        "instance_id": instance_id,
    }

    ctx = mp.get_context('spawn')
    draft_process = ctx.Process(
        target=_draft_process_entry,
        args=(child_conn, sdw._draft_worker_kwargs,
              sdw._num_gpu_blocks, sdw._num_cpu_blocks,
              logit_buffer_config),
        name="cospec-draft-worker",
    )
    draft_process.start()
    sdw._draft_process = draft_process

    logger.info("CoSpec: draft worker process spawned (PID: %d)",
                 draft_process.pid)

    # Wait for draft process to signal ready (or error).
    # Timeout prevents infinite hang if draft segfaults during CUDA init.
    _DRAFT_READY_TIMEOUT_S = 120
    if not parent_conn.poll(timeout=_DRAFT_READY_TIMEOUT_S):
        draft_process.kill()
        draft_process.join(timeout=5)
        raise RuntimeError(
            f"Draft worker process (PID: {draft_process.pid}) did not send "
            f"READY signal within {_DRAFT_READY_TIMEOUT_S}s. "
            "It may have crashed during CUDA initialization.")
    signal, payload = parent_conn.recv()
    if signal == "ERROR":
        raise RuntimeError(
            f"Draft worker process failed to start:\n{payload}")
    assert signal == "READY", f"Unexpected signal: {signal}"
    logger.info("CoSpec: draft worker process ready (PID: %d)", payload)

    # Create RPC client and orchestrator
    draft_rpc = DraftWorkerRPC(parent_conn)
    target_sm_ratio = envs.COSPEC_TARGET_SM_RATIO
    logger.info("CoSpec: using target SM ratio %.2f", target_sm_ratio)

    sdw.orchestrator = CoSpecOrchestrator(
        spec_decode_worker=sdw,
        draft_rpc=draft_rpc,
        sm_controller=sdw.sm_controller,
        target_sm_ratio=target_sm_ratio,
        shared_logit_buffer=shared_logit_buffer,
    )
    logger.info("CoSpec: orchestrator created")

    # Register for cleanup at interpreter exit (more reliable than __del__)
    _register_for_cleanup(sdw)


def cleanup_cospec(sdw: Any) -> None:
    """Clean up CoSpec draft process. Called by __del__ and atexit."""
    if getattr(sdw, '_cospec_cleaned_up', False):
        return
    sdw._cospec_cleaned_up = True

    # Shutdown orchestrator (sends SHUTDOWN to draft RPC)
    if getattr(sdw, 'orchestrator', None) is not None:
        try:
            sdw.orchestrator.shutdown()
        except Exception:
            pass
        sdw.orchestrator = None

    # Terminate draft process
    if getattr(sdw, '_draft_process', None) is not None:
        try:
            sdw._draft_process.join(timeout=2.0)
            if sdw._draft_process.is_alive():
                sdw._draft_process.terminate()
                sdw._draft_process.join(timeout=2.0)
            if sdw._draft_process.is_alive():
                sdw._draft_process.kill()
                sdw._draft_process.join(timeout=1.0)
        except Exception:
            pass
        sdw._draft_process = None

    # Clean up stale IPC handles
    try:
        cleanup_cospec_resources()
    except Exception:
        pass


def _register_for_cleanup(sdw: Any) -> None:
    """Register a SpecDecodeWorker for atexit cleanup."""
    global _atexit_registered, _cospec_instances
    _cospec_instances.append(weakref.ref(sdw))
    if not _atexit_registered:
        atexit.register(_atexit_cleanup)
        _atexit_registered = True


def _atexit_cleanup() -> None:
    """Clean up all CoSpec instances at interpreter exit."""
    for ref in _cospec_instances:
        worker = ref()
        if worker is not None:
            cleanup_cospec(worker)


def _draft_process_entry(
    child_conn,
    draft_worker_kwargs: dict,
    num_gpu_blocks: int,
    num_cpu_blocks: int,
    logit_buffer_config: Optional[dict] = None,
) -> None:
    """Entry point for the draft worker process.

    This runs in a new process with its own CUDA context, enabling true
    MPS SM-partitioned concurrency with the target process.
    """
    try:
        import torch

        from vllm.cospec.sm_controller import SMController
        from vllm.cospec.worker_rpc import DraftWorkerServer
        from vllm.spec_decode.multi_step_worker import MultiStepWorker

        # Initialize CUDA in the new process
        device = draft_worker_kwargs['vllm_config'].device_config.device
        if device.index is None:
            device = torch.device(device.type, 0)
        torch.cuda.set_device(device)

        # Create draft worker (loads model weights from CUDA IPC handles
        # via SharedMemoryModelLoader when handles file exists)
        proposer_worker = MultiStepWorker(**draft_worker_kwargs)

        # Init distributed env on a different port to avoid conflict
        # with the target process's torch.distributed
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('', 0))
        free_port = sock.getsockname()[1]
        sock.close()

        vllm_config = draft_worker_kwargs['vllm_config']
        # Clear static forward context — it was pickled from the target
        # process and contains stale attention layer references
        vllm_config.compilation_config.static_forward_context.clear()

        proposer_worker.worker.distributed_init_method = \
            f"tcp://localhost:{free_port}"
        proposer_worker.init_device()
        proposer_worker.load_model()

        # Initialize KV cache (draft has its own, separate from target)
        proposer_worker.initialize_cache(
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )

        # Configure sampler for speculative decoding
        proposer_worker.set_include_gpu_probs_tensor()
        proposer_worker.set_should_modify_greedy_probs_inplace()

        # Create SM controller for draft process
        sm_controller = SMController(is_target=False)

        # Open shared logit buffer (client side)
        shared_logit_buffer = None
        if logit_buffer_config is not None:
            from vllm.cospec.shared_memory import SharedLogitBuffer
            shared_logit_buffer = SharedLogitBuffer(
                mode="client", **logit_buffer_config)

        # Start server event loop (blocks until shutdown)
        server = DraftWorkerServer(
            conn=child_conn,
            draft_worker=proposer_worker,
            sm_controller=sm_controller,
            shared_logit_buffer=shared_logit_buffer,
        )

        # Signal parent that we're ready
        logger.info("CoSpec draft process: ready (PID: %d)", os.getpid())
        child_conn.send(("READY", os.getpid()))
        server.serve()

    except Exception:
        traceback.print_exc()
        try:
            child_conn.send(("ERROR", traceback.format_exc()))
        except Exception:
            pass
