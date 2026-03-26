"""
MAX Serve Benchmark Server Lifecycle

Manages MAX serve processes: start, health-check, GPU memory snapshot, stop.
"""

import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import aiohttp

from benchmark_config import MAXBenchmarkConfig


async def start_servers(config: MAXBenchmarkConfig, log_dir: Path | None = None) -> list[subprocess.Popen]:
    """
    Start MAX serve processes for the given config.

    Each server gets its own device slice and port.
    Returns list of Popen processes.
    """
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["HF_HOME"] = "/opt/dlami/nvme/huggingface"

    processes = []
    for i in range(config.num_servers):
        port = config.server_port(i)
        args = config.max_serve_args(server_index=i)

        log_path = log_dir / f"{config.config_id}_server_{i}.log" if log_dir else None
        log_file = open(log_path, "w") if log_path else subprocess.DEVNULL

        print(f"  Starting server {i}: port={port}, devices={config.device_string(i)}")
        proc = subprocess.Popen(
            args,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        processes.append(proc)

        # Stagger startup to avoid simultaneous GPU memory allocation
        if i < config.num_servers - 1:
            import time as _time
            _time.sleep(3)

    return processes


async def wait_for_healthy(
    config: MAXBenchmarkConfig,
    timeout: float = 600.0,
    poll_interval: float = 5.0,
) -> list[str]:
    """
    Poll /health on all expected servers until all are healthy or timeout.

    Returns list of healthy server URLs.
    Raises TimeoutError if not all servers are healthy within timeout.
    """
    urls = config.server_urls()
    start = time.monotonic()
    healthy_set = set()

    while time.monotonic() - start < timeout:
        async with aiohttp.ClientSession() as session:
            for url in urls:
                if url in healthy_set:
                    continue
                try:
                    async with session.get(
                        f"{url}/health",
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        if resp.status == 200:
                            healthy_set.add(url)
                            print(f"  Server {url} healthy ({len(healthy_set)}/{len(urls)})")
                except Exception:
                    pass

        if len(healthy_set) == len(urls):
            return list(sorted(healthy_set))

        elapsed = time.monotonic() - start
        print(f"  Waiting for servers... {len(healthy_set)}/{len(urls)} healthy ({elapsed:.0f}s)")
        await asyncio.sleep(poll_interval)

    # Timeout — return whatever is healthy
    if not healthy_set:
        raise TimeoutError(
            f"No servers became healthy within {timeout}s for config {config.config_id}"
        )

    print(f"  WARNING: Only {len(healthy_set)}/{len(urls)} servers healthy after {timeout}s")
    return list(sorted(healthy_set))


def stop_servers(processes: list[subprocess.Popen], graceful_timeout: float = 10.0):
    """Stop all server processes. SIGTERM first, SIGKILL after timeout."""
    # Send SIGTERM to all
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                pass

    # Wait for graceful shutdown
    deadline = time.monotonic() + graceful_timeout
    for proc in processes:
        remaining = max(0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            pass

    # Force kill any still running
    for proc in processes:
        if proc.poll() is None:
            try:
                proc.kill()
                proc.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass

    print(f"  All {len(processes)} server processes stopped")


def snapshot_gpu_memory() -> dict:
    """Query nvidia-smi for per-GPU memory usage."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                gpus.append({
                    "index": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "gpu_utilization_pct": int(parts[4]),
                })

        total_used = sum(g["memory_used_mb"] for g in gpus)
        total_capacity = sum(g["memory_total_mb"] for g in gpus)

        return {
            "gpus": gpus,
            "total_used_mb": total_used,
            "total_capacity_mb": total_capacity,
            "utilization_pct": round(total_used / total_capacity * 100, 1) if total_capacity else 0,
        }
    except Exception as e:
        return {"error": str(e)}
