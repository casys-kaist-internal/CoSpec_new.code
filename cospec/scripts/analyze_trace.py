#!/usr/bin/env python3
"""Analyze CoSpec profiler trace to identify bottlenecks.

Reads a Chrome trace JSON (from torch.profiler) and extracts timing
for cospec:: annotated regions, CUDA kernels, and CPU ops.
"""

import json
import sys
from collections import defaultdict


def analyze_trace(trace_path: str):
    print(f"Loading trace from {trace_path}...")
    with open(trace_path, "r") as f:
        data = json.load(f)

    events = data if isinstance(data, list) else data.get("traceEvents", [])
    print(f"Total events: {len(events)}")

    # ── 1. CoSpec annotated regions ──
    cospec_events = defaultdict(list)  # label -> [duration_us]
    for ev in events:
        name = ev.get("name", "")
        if name.startswith("cospec::") and ev.get("ph") == "X":
            dur = ev.get("dur", 0)
            cospec_events[name].append(dur)

    if cospec_events:
        print("\n" + "="*70)
        print("CoSpec Annotated Regions (CPU wall time)")
        print("="*70)
        print(f"{'Label':<45} {'Count':>6} {'Mean ms':>9} {'Med ms':>9} "
              f"{'P95 ms':>9} {'Total ms':>10}")
        print("-"*90)

        sorted_labels = sorted(cospec_events.keys(),
                               key=lambda k: sum(cospec_events[k]),
                               reverse=True)
        for label in sorted_labels:
            durs = sorted(cospec_events[label])
            count = len(durs)
            mean_us = sum(durs) / count
            med_us = durs[count // 2]
            p95_us = durs[int(count * 0.95)]
            total_us = sum(durs)
            print(f"{label:<45} {count:>6} {mean_us/1000:>9.2f} "
                  f"{med_us/1000:>9.2f} {p95_us/1000:>9.2f} "
                  f"{total_us/1000:>10.1f}")
    else:
        print("\nNo cospec:: annotated events found!")

    # ── 2. Top CUDA kernels by total time ──
    cuda_kernels = defaultdict(list)  # kernel_name -> [duration_us]
    for ev in events:
        cat = ev.get("cat", "")
        if cat == "kernel" and ev.get("ph") == "X":
            name = ev.get("name", "unknown")
            dur = ev.get("dur", 0)
            cuda_kernels[name].append(dur)

    if cuda_kernels:
        print("\n" + "="*70)
        print("Top 20 CUDA Kernels by Total GPU Time")
        print("="*70)
        print(f"{'Kernel':<60} {'Count':>6} {'Mean us':>9} {'Total ms':>10} "
              f"{'%':>6}")
        print("-"*95)

        total_gpu_time = sum(sum(v) for v in cuda_kernels.values())
        sorted_kernels = sorted(cuda_kernels.items(),
                                key=lambda kv: sum(kv[1]),
                                reverse=True)
        for name, durs in sorted_kernels[:20]:
            count = len(durs)
            mean_us = sum(durs) / count
            total_us = sum(durs)
            pct = total_us / total_gpu_time * 100 if total_gpu_time > 0 else 0
            # Truncate long kernel names
            short_name = name[:57] + "..." if len(name) > 60 else name
            print(f"{short_name:<60} {count:>6} {mean_us:>9.1f} "
                  f"{total_us/1000:>10.1f} {pct:>5.1f}%")

        print(f"\nTotal GPU kernel time: {total_gpu_time/1000:.1f} ms")

    # ── 3. Top CPU ops by total time ──
    cpu_ops = defaultdict(list)
    for ev in events:
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if (ev.get("ph") == "X" and cat in ("cpu_op", "user_annotation")
                and not name.startswith("cospec::")
                and not name.startswith("ProfilerStep")):
            dur = ev.get("dur", 0)
            cpu_ops[name].append(dur)

    if cpu_ops:
        print("\n" + "="*70)
        print("Top 20 CPU Ops by Total Time")
        print("="*70)
        print(f"{'Op':<55} {'Count':>7} {'Mean ms':>9} {'Total ms':>10}")
        print("-"*85)

        sorted_ops = sorted(cpu_ops.items(),
                            key=lambda kv: sum(kv[1]),
                            reverse=True)
        for name, durs in sorted_ops[:20]:
            count = len(durs)
            mean_us = sum(durs) / count
            total_us = sum(durs)
            short_name = name[:52] + "..." if len(name) > 55 else name
            print(f"{short_name:<55} {count:>7} {mean_us/1000:>9.2f} "
                  f"{total_us/1000:>10.1f}")

    # ── 4. CUDA Runtime API calls (sync, memcpy, etc.) ──
    cuda_runtime = defaultdict(list)
    for ev in events:
        cat = ev.get("cat", "")
        if cat == "cuda_runtime" and ev.get("ph") == "X":
            name = ev.get("name", "")
            dur = ev.get("dur", 0)
            cuda_runtime[name].append(dur)

    if cuda_runtime:
        print("\n" + "="*70)
        print("CUDA Runtime API Calls")
        print("="*70)
        print(f"{'Call':<45} {'Count':>7} {'Mean ms':>9} {'Total ms':>10}")
        print("-"*75)

        sorted_runtime = sorted(cuda_runtime.items(),
                                key=lambda kv: sum(kv[1]),
                                reverse=True)
        for name, durs in sorted_runtime[:15]:
            count = len(durs)
            mean_us = sum(durs) / count
            total_us = sum(durs)
            print(f"{name:<45} {count:>7} {mean_us/1000:>9.2f} "
                  f"{total_us/1000:>10.1f}")

    # ── 5. Per-step timing analysis ──
    # Find all cospec::step events and analyze their sub-regions
    step_events = [ev for ev in events
                   if ev.get("name") == "cospec::step" and ev.get("ph") == "X"]
    if step_events:
        print("\n" + "="*70)
        print(f"Per-Step Analysis ({len(step_events)} concurrent steps)")
        print("="*70)

        step_durs = [ev["dur"] for ev in step_events]
        step_durs.sort()
        print(f"Step duration: mean={sum(step_durs)/len(step_durs)/1000:.2f}ms "
              f"med={step_durs[len(step_durs)//2]/1000:.2f}ms "
              f"p95={step_durs[int(len(step_durs)*0.95)]/1000:.2f}ms "
              f"min={step_durs[0]/1000:.2f}ms max={step_durs[-1]/1000:.2f}ms")

    bootstrap_events = [ev for ev in events
                        if ev.get("name") == "cospec::bootstrap_step"
                        and ev.get("ph") == "X"]
    if bootstrap_events:
        bs_durs = [ev["dur"] for ev in bootstrap_events]
        bs_durs.sort()
        print(f"\nBootstrap steps: {len(bootstrap_events)} events")
        print(f"Bootstrap duration: mean={sum(bs_durs)/len(bs_durs)/1000:.2f}ms "
              f"med={bs_durs[len(bs_durs)//2]/1000:.2f}ms "
              f"max={bs_durs[-1]/1000:.2f}ms")

    # ── 6. Breakdown: verification vs draft wait ──
    verify_durs = cospec_events.get("cospec::run_verification", [])
    collect_durs = cospec_events.get("cospec::propose_collect_recv", [])
    if verify_durs and collect_durs:
        print("\n" + "="*70)
        print("Draft vs Target Overlap Analysis")
        print("="*70)
        avg_verify = sum(verify_durs) / len(verify_durs) / 1000
        avg_collect = sum(collect_durs) / len(collect_durs) / 1000
        print(f"Avg target verify: {avg_verify:.2f}ms")
        print(f"Avg draft collect (blocking wait): {avg_collect:.2f}ms")
        if avg_collect > 0.5:
            print(f"  → Draft is SLOWER than target by ~{avg_collect:.1f}ms "
                  "(target finishes first, then waits for draft)")
        else:
            print(f"  → Draft finishes before target (good overlap)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/cospec_trace_annotated.json"
    analyze_trace(path)
