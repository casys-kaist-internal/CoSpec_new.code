#!/usr/bin/env python3
"""Deep analysis: identify specific optimization opportunities."""

import json
import sys
from collections import defaultdict


def analyze(path):
    with open(path) as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])

    # ── 1. Bootstrap vs Concurrent step ratio ──
    bootstrap_events = [e for e in events
                        if e.get("name") == "cospec::bootstrap_step"
                        and e.get("ph") == "X"]
    concurrent_events = [e for e in events
                         if e.get("name") == "cospec::step"
                         and e.get("ph") == "X"
                         and e.get("dur", 0) > 1000]  # >1ms

    total_bootstrap_ms = sum(e["dur"] for e in bootstrap_events) / 1000
    total_concurrent_ms = sum(e["dur"] for e in concurrent_events) / 1000
    total_ms = total_bootstrap_ms + total_concurrent_ms

    print("="*70)
    print("Step Type Distribution")
    print("="*70)
    print(f"Bootstrap steps: {len(bootstrap_events)} "
          f"({total_bootstrap_ms:.0f}ms, {total_bootstrap_ms/total_ms*100:.1f}%)")
    print(f"Concurrent steps: {len(concurrent_events)} "
          f"({total_concurrent_ms:.0f}ms, {total_concurrent_ms/total_ms*100:.1f}%)")
    print(f"Total profiled time: {total_ms:.0f}ms")

    # ── 2. Score proposals: CUDA graph vs eager analysis ──
    # Look for cudaGraphLaunch vs cudaLaunchKernel inside score_proposals
    score_events = [e for e in events
                    if e.get("name") == "cospec::score_proposals"
                    and e.get("ph") == "X"]

    print(f"\n{'='*70}")
    print("Score Proposals Analysis")
    print(f"{'='*70}")
    print(f"Total calls: {len(score_events)}")

    if score_events:
        durs = sorted([e["dur"]/1000 for e in score_events])
        n = len(durs)
        # Split into "fast" (CUDA graph) and "slow" (eager)
        fast = [d for d in durs if d < 100]
        slow = [d for d in durs if d >= 100]
        print(f"  Fast (<100ms): {len(fast)} calls, "
              f"mean={sum(fast)/len(fast):.1f}ms" if fast else "  Fast: 0")
        print(f"  Slow (>=100ms): {len(slow)} calls, "
              f"mean={sum(slow)/len(slow):.1f}ms" if slow else "  Slow: 0")

    # ── 3. index_put_ hotspot analysis ──
    print(f"\n{'='*70}")
    print("aten::index_put_ Spike Analysis")
    print(f"{'='*70}")

    iput = [e for e in events
            if e.get("name") == "aten::index_put_" and e.get("ph") == "X"]
    if iput:
        durs = sorted([e["dur"]/1000 for e in iput])
        fast = [d for d in durs if d < 1]
        medium = [d for d in durs if 1 <= d < 50]
        slow = [d for d in durs if d >= 50]
        print(f"  <1ms: {len(fast)} calls")
        print(f"  1-50ms: {len(medium)} calls"
              + (f" (mean={sum(medium)/len(medium):.1f}ms)" if medium else ""))
        print(f"  >=50ms: {len(slow)} calls"
              + (f" (mean={sum(slow)/len(slow):.1f}ms, "
                 f"max={max(slow):.1f}ms)" if slow else ""))
        # Total time in spikes
        spike_total = sum(slow) + sum(medium)
        print(f"  Time in spikes (>=1ms): {spike_total:.1f}ms")

    # ── 4. Memory allocation analysis (cudaMalloc during serving) ──
    print(f"\n{'='*70}")
    print("Memory Allocation During Serving")
    print(f"{'='*70}")

    mallocs = [e for e in events
               if e.get("name") == "cudaMalloc" and e.get("ph") == "X"]
    if mallocs:
        durs = sorted([e["dur"]/1000 for e in mallocs])
        print(f"  cudaMalloc calls: {len(mallocs)}")
        print(f"  Total time: {sum(durs):.1f}ms")
        print(f"  Mean: {sum(durs)/len(durs):.2f}ms")
        print(f"  Max: {max(durs):.2f}ms")

    # ── 5. verify_tokens breakdown ──
    print(f"\n{'='*70}")
    print("verify_tokens Analysis")
    print(f"{'='*70}")

    verify = [e for e in events
              if e.get("name") == "cospec::verify_tokens" and e.get("ph") == "X"]
    if verify:
        durs = sorted([e["dur"]/1000 for e in verify])
        print(f"  Calls: {len(verify)}")
        print(f"  Mean: {sum(durs)/len(durs):.2f}ms")
        print(f"  P95: {durs[int(len(durs)*0.95)]:.2f}ms")
        print(f"  Max: {max(durs):.2f}ms")

    # ── 6. CPU time between GPU-bound operations ──
    # Look for gaps between CUDA kernel launches
    print(f"\n{'='*70}")
    print("CUDA Graph Usage")
    print(f"{'='*70}")

    graph_launches = [e for e in events
                      if e.get("name") == "cudaGraphLaunch"
                      and e.get("ph") == "X"]
    kernel_launches = [e for e in events
                       if e.get("name") == "cudaLaunchKernel"
                       and e.get("ph") == "X"]
    print(f"  cudaGraphLaunch: {len(graph_launches)} calls"
          + (f" (total={sum(e['dur'] for e in graph_launches)/1000:.1f}ms)"
             if graph_launches else ""))
    print(f"  cudaLaunchKernel: {len(kernel_launches)} calls"
          + (f" (total={sum(e['dur'] for e in kernel_launches)/1000:.1f}ms)"
             if kernel_launches else ""))

    # ── 7. Largest single-event bottlenecks ──
    print(f"\n{'='*70}")
    print("Top 10 Longest Individual Events")
    print(f"{'='*70}")

    all_x = [(e.get("name", "?"), e.get("cat", "?"), e["dur"]/1000)
             for e in events if e.get("ph") == "X" and e.get("dur", 0) > 0]
    all_x.sort(key=lambda x: x[2], reverse=True)
    for name, cat, dur_ms in all_x[:10]:
        short = name[:55] + "..." if len(name) > 55 else name
        print(f"  {dur_ms:>8.1f}ms  [{cat}]  {short}")

    # ── 8. Propose timing in bootstrap (where it's blocking) ──
    print(f"\n{'='*70}")
    print("Bootstrap Propose (Blocking Draft) Analysis")
    print(f"{'='*70}")

    bp = [e for e in events
          if e.get("name") == "cospec::bootstrap_propose"
          and e.get("ph") == "X"]
    if bp:
        durs = sorted([e["dur"]/1000 for e in bp])
        print(f"  Calls: {len(bp)}")
        print(f"  Mean: {sum(durs)/len(durs):.2f}ms")
        print(f"  Median: {durs[len(durs)//2]:.2f}ms")
        print(f"  P95: {durs[int(len(durs)*0.95)]:.2f}ms")
        print(f"  Total: {sum(durs):.1f}ms")
        print(f"  This is BLOCKING time where target GPU is idle!")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/cospec_trace_annotated.json"
    analyze(path)
