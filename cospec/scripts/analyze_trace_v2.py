#!/usr/bin/env python3
"""Deeper analysis of CoSpec trace: per-step breakdown, overlap quality,
and identification of specific bottleneck patterns."""

import json
import sys
from collections import defaultdict


def analyze_trace(trace_path: str):
    print(f"Loading trace from {trace_path}...")
    with open(trace_path, "r") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data.get("traceEvents", [])

    # Find thread IDs for the main thread (where cospec:: events fire)
    cospec_tids = set()
    for ev in events:
        if ev.get("name", "").startswith("cospec::") and ev.get("ph") == "X":
            cospec_tids.add(ev.get("tid"))

    # ── 1. Detailed step-level breakdown ──
    # For each cospec::step, find all nested cospec:: regions
    step_events = []
    for ev in events:
        if ev.get("name") == "cospec::step" and ev.get("ph") == "X":
            step_events.append(ev)

    if not step_events:
        print("No cospec::step events found!")
        return

    # Build index of all cospec:: events by timestamp for nesting analysis
    all_cospec = []
    for ev in events:
        if (ev.get("name", "").startswith("cospec::")
                and ev.get("ph") == "X"):
            all_cospec.append(ev)
    all_cospec.sort(key=lambda e: e.get("ts", 0))

    # ── 2. Analyze concurrent steps (the interesting ones) ──
    print("\n" + "="*70)
    print("Concurrent Step Breakdown (cospec::step events)")
    print("="*70)

    # For each step, find its sub-regions
    sub_region_times = defaultdict(list)  # region_name -> [dur_ms]
    step_durations = []

    for step in step_events:
        step_ts = step["ts"]
        step_end = step_ts + step["dur"]
        step_dur_ms = step["dur"] / 1000

        # Skip very short steps (< 1ms, likely edge cases)
        if step_dur_ms < 1.0:
            continue

        step_durations.append(step_dur_ms)

        # Find all sub-regions within this step
        for sub in all_cospec:
            if (sub["ts"] >= step_ts and sub["ts"] + sub["dur"] <= step_end
                    and sub["name"] != "cospec::step"):
                sub_region_times[sub["name"]].append(sub["dur"] / 1000)

    if step_durations:
        step_durations.sort()
        n = len(step_durations)
        print(f"\nConcurrent steps (>1ms): {n}")
        print(f"  Mean: {sum(step_durations)/n:.1f}ms")
        print(f"  Median: {step_durations[n//2]:.1f}ms")
        print(f"  P95: {step_durations[int(n*0.95)]:.1f}ms")
        print(f"  Max: {step_durations[-1]:.1f}ms")

        print(f"\nSub-region breakdown (avg ms across {n} steps):")
        print(f"{'Region':<45} {'Mean ms':>9} {'Med ms':>9} {'P95 ms':>9} {'%step':>7}")
        print("-"*85)

        avg_step = sum(step_durations) / n
        sorted_regions = sorted(sub_region_times.items(),
                                key=lambda kv: sum(kv[1]) / len(kv[1]),
                                reverse=True)
        for name, durs in sorted_regions:
            durs_sorted = sorted(durs)
            m = len(durs_sorted)
            mean = sum(durs_sorted) / m
            med = durs_sorted[m // 2]
            p95 = durs_sorted[int(m * 0.95)]
            pct = mean / avg_step * 100 if avg_step > 0 else 0
            short = name.replace("cospec::", "")
            print(f"  {short:<43} {mean:>9.2f} {med:>9.2f} "
                  f"{p95:>9.2f} {pct:>6.1f}%")

    # ── 3. Score_proposals breakdown ──
    # score_proposals is the biggest — what's inside it?
    print("\n" + "="*70)
    print("Inside cospec::score_proposals")
    print("="*70)

    score_events = [ev for ev in events
                    if ev.get("name") == "cospec::score_proposals"
                    and ev.get("ph") == "X"]

    # Look for aten:: ops that overlap with score_proposals
    score_inner_ops = defaultdict(list)
    for score_ev in score_events:
        s_ts = score_ev["ts"]
        s_end = s_ts + score_ev["dur"]
        for ev in events:
            if (ev.get("ph") == "X"
                    and ev.get("ts", 0) >= s_ts
                    and ev.get("ts", 0) + ev.get("dur", 0) <= s_end
                    and ev.get("cat") in ("cpu_op",)
                    and not ev.get("name", "").startswith("cospec::")):
                score_inner_ops[ev["name"]].append(ev["dur"] / 1000)

    if score_inner_ops:
        sorted_inner = sorted(score_inner_ops.items(),
                              key=lambda kv: sum(kv[1]),
                              reverse=True)
        print(f"{'Op':<45} {'Count':>7} {'Mean ms':>9} {'Total ms':>10}")
        print("-"*75)
        for name, durs in sorted_inner[:15]:
            print(f"  {name:<43} {len(durs):>7} {sum(durs)/len(durs):>9.2f} "
                  f"{sum(durs):>10.1f}")

    # ── 4. cudaStreamSynchronize analysis ──
    print("\n" + "="*70)
    print("cudaStreamSynchronize Analysis")
    print("="*70)

    sync_events = [ev for ev in events
                   if ev.get("name") == "cudaStreamSynchronize"
                   and ev.get("ph") == "X"]
    if sync_events:
        sync_durs = sorted([ev["dur"] / 1000 for ev in sync_events])
        n = len(sync_durs)
        print(f"Total calls: {n}")
        print(f"  Mean: {sum(sync_durs)/n:.2f}ms")
        print(f"  Median: {sync_durs[n//2]:.2f}ms")
        print(f"  P95: {sync_durs[int(n*0.95)]:.2f}ms")
        print(f"  Max: {sync_durs[-1]:.2f}ms")
        print(f"  Total: {sum(sync_durs):.1f}ms")

        # Distribution
        bins = [0, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, float('inf')]
        labels = ["<0.01ms", "0.01-0.1ms", "0.1-1ms", "1-10ms",
                  "10-50ms", "50-100ms", ">100ms"]
        counts = [0] * len(labels)
        for d in sync_durs:
            for i in range(len(bins) - 1):
                if bins[i] <= d < bins[i+1]:
                    counts[i] += 1
                    break
        print("\nDistribution:")
        for label, count in zip(labels, counts):
            if count > 0:
                print(f"  {label:<15} {count:>5} ({count/n*100:.1f}%)")

    # ── 5. aten::index_put_ analysis ──
    print("\n" + "="*70)
    print("aten::index_put_ Analysis (KV cache updates)")
    print("="*70)
    iput_events = [ev for ev in events
                   if ev.get("name") == "aten::index_put_"
                   and ev.get("ph") == "X"]
    if iput_events:
        iput_durs = sorted([ev["dur"] / 1000 for ev in iput_events])
        n = len(iput_durs)
        print(f"Total calls: {n}")
        print(f"  Mean: {sum(iput_durs)/n:.2f}ms")
        print(f"  Median: {iput_durs[n//2]:.2f}ms")
        print(f"  P95: {iput_durs[int(n*0.95)]:.2f}ms")
        print(f"  Max: {iput_durs[-1]:.2f}ms")
        print(f"  Total: {sum(iput_durs):.1f}ms")

    # ── 6. Overlap quality: for each concurrent step, how much time
    #    does the orchestrator WAIT after run_verification finishes? ──
    print("\n" + "="*70)
    print("Overlap Quality per Concurrent Step")
    print("="*70)

    verify_events = {ev["ts"]: ev for ev in events
                     if ev.get("name") == "cospec::run_verification"
                     and ev.get("ph") == "X"}
    collect_events = {ev["ts"]: ev for ev in events
                      if ev.get("name") == "cospec::propose_collect_recv"
                      and ev.get("ph") == "X"}

    # For each step, find its verify and collect events
    wait_times = []
    for step in step_events:
        s_ts = step["ts"]
        s_end = s_ts + step["dur"]
        if step["dur"] / 1000 < 1.0:
            continue

        step_verify = None
        step_collect = None
        for ev in all_cospec:
            if (ev["ts"] >= s_ts and ev["ts"] + ev["dur"] <= s_end):
                if ev["name"] == "cospec::run_verification":
                    step_verify = ev
                elif ev["name"] == "cospec::propose_collect_recv":
                    step_collect = ev

        if step_verify and step_collect:
            # Wait time = collect duration (how long we block waiting for draft)
            wait_ms = step_collect["dur"] / 1000
            verify_ms = step_verify["dur"] / 1000
            # "wasted" time = step_total - verify - split_batch overhead
            step_ms = step["dur"] / 1000
            overhead_ms = step_ms - verify_ms - wait_ms
            wait_times.append({
                "step_ms": step_ms,
                "verify_ms": verify_ms,
                "wait_ms": wait_ms,
                "overhead_ms": overhead_ms,
            })

    if wait_times:
        n = len(wait_times)
        avg_step = sum(w["step_ms"] for w in wait_times) / n
        avg_verify = sum(w["verify_ms"] for w in wait_times) / n
        avg_wait = sum(w["wait_ms"] for w in wait_times) / n
        avg_overhead = sum(w["overhead_ms"] for w in wait_times) / n

        print(f"Concurrent steps analyzed: {n}")
        print(f"  Avg step total: {avg_step:.1f}ms")
        print(f"  Avg verify (target): {avg_verify:.1f}ms")
        print(f"  Avg draft wait: {avg_wait:.1f}ms")
        print(f"  Avg overhead: {avg_overhead:.1f}ms")
        print(f"  Overlap efficiency: {(1 - avg_wait/avg_step)*100:.1f}% "
              f"(100% = draft finishes before target)")

        # Show worst 5 steps
        worst = sorted(wait_times, key=lambda w: w["wait_ms"], reverse=True)
        print(f"\nWorst 5 draft-wait steps:")
        for w in worst[:5]:
            print(f"  step={w['step_ms']:.1f}ms "
                  f"verify={w['verify_ms']:.1f}ms "
                  f"wait={w['wait_ms']:.1f}ms")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/workspace/cospec_trace_annotated.json"
    analyze_trace(path)
