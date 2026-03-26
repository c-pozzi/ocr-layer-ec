"""
MAX Serve Benchmark Configuration Matrix

Defines all MAX serve configurations to test on A100 GPUs.
Each config specifies model, quantization, device assignment, and batching params.
"""

from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "benchmark_a100"

# Base models (NOT AWQ — MAX does its own quantization)
MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_32B = "Qwen/Qwen2.5-VL-32B-Instruct"

# Cost per hour for p4d.24xlarge (on-demand)
P4D_COST_PER_HOUR = 32.77


@dataclass
class MAXBenchmarkConfig:
    config_id: str
    model: str
    gpus_per_server: int  # 1, 2, or 4
    max_batch_size: int
    quantization_encoding: str = "bfloat16"  # Qwen2.5-VL only supports bfloat16 on MAX
    device_memory_utilization: float = 0.85  # Needs headroom for vision encoder activations
    max_length: int = 8192
    max_batch_context_length: int | None = None
    cache_strategy: str = "model_default"  # "model_default" or "paged"

    # Benchmark parameters
    workloads: list[str] = field(default_factory=list)
    pages_per_workload: int = 200
    num_runs: int = 3
    concurrency_per_server: int = 4  # Lower than vLLM — vision tokens use more memory per request
    warmup_requests: int = 10

    @property
    def num_servers(self) -> int:
        return 8 // self.gpus_per_server

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.config_id

    def device_string(self, server_index: int) -> str:
        """Return --devices string for a given server index, e.g. 'gpu:0' or 'gpu:0,1'."""
        start = server_index * self.gpus_per_server
        ids = list(range(start, start + self.gpus_per_server))
        return "gpu:" + ",".join(str(d) for d in ids)

    def max_serve_args(self, server_index: int) -> list[str]:
        """Generate max serve CLI arguments for one server instance."""
        port = self.server_port(server_index)
        args = [
            "max", "serve",
            "--model-path", self.model,
            "--max-length", str(self.max_length),
            "--devices", self.device_string(server_index),
            "--port", str(port),
        ]
        # Qwen2.5-VL only supports bfloat16 — skip --quantization-encoding
        args.extend(["--device-memory-utilization", str(self.device_memory_utilization)])
        if self.max_batch_size:
            args.extend(["--max-batch-size", str(self.max_batch_size)])
        if self.max_batch_context_length is not None:
            args.extend(["--max-batch-context-length", str(self.max_batch_context_length)])
        if self.cache_strategy != "model_default":
            args.extend(["--cache-strategy", self.cache_strategy])
        return args

    def server_port(self, server_index: int) -> int:
        """Return API port for a given server. Spaced by 10 to avoid metrics port conflicts."""
        return 8000 + server_index * 10

    def server_urls(self) -> list[str]:
        """Return list of server URLs."""
        return [f"http://localhost:{self.server_port(i)}" for i in range(self.num_servers)]


# =============================================================================
# 7B CONFIGS
# =============================================================================

def get_7b_configs() -> list[MAXBenchmarkConfig]:
    """Return all 7B benchmark configurations (bfloat16 only for Qwen2.5-VL).

    Vision models need small batch sizes on 40GB A100 — image tokens are large.
    Tested: max_batch_size=8 + max_length=4096 works reliably.
    """
    workloads = ["classify", "simple_ocr"]

    configs = [
        # Conservative baseline (proven to work)
        MAXBenchmarkConfig(
            config_id="max_7b_bs8",
            model=MODEL_7B, gpus_per_server=1, max_batch_size=8,
            max_length=4096,
            workloads=workloads,
        ),
        # Step up batch size
        MAXBenchmarkConfig(
            config_id="max_7b_bs16",
            model=MODEL_7B, gpus_per_server=1, max_batch_size=16,
            max_length=4096,
            workloads=workloads,
        ),
        MAXBenchmarkConfig(
            config_id="max_7b_bs32",
            model=MODEL_7B, gpus_per_server=1, max_batch_size=32,
            max_length=4096,
            workloads=workloads,
        ),
        # Full context (8192) with small batch
        MAXBenchmarkConfig(
            config_id="max_7b_bs8_ctx8k",
            model=MODEL_7B, gpus_per_server=1, max_batch_size=8,
            max_length=8192,
            workloads=workloads,
        ),
        MAXBenchmarkConfig(
            config_id="max_7b_bs16_ctx8k",
            model=MODEL_7B, gpus_per_server=1, max_batch_size=16,
            max_length=8192,
            workloads=workloads,
        ),
        # Paged cache
        MAXBenchmarkConfig(
            config_id="max_7b_bs16_paged",
            model=MODEL_7B, gpus_per_server=1, max_batch_size=16,
            max_length=4096, cache_strategy="paged",
            workloads=workloads,
        ),
    ]
    return configs


# =============================================================================
# 32B CONFIGS (bfloat16 ~65GB — needs 2+ GPUs on 40GB A100)
# =============================================================================

def get_32b_configs() -> list[MAXBenchmarkConfig]:
    """Return all 32B benchmark configurations (bfloat16 only).

    32B bf16 needs minimum 2x 40GB GPUs. Small batches for vision.
    """
    workloads = ["complex_ocr"]

    configs = [
        # 2-GPU configs (4 servers) — tight memory
        MAXBenchmarkConfig(
            config_id="max_32b_2gpu_bs4",
            model=MODEL_32B, gpus_per_server=2, max_batch_size=4,
            max_length=8192,
            pages_per_workload=100,
            workloads=workloads,
        ),
        MAXBenchmarkConfig(
            config_id="max_32b_2gpu_bs8",
            model=MODEL_32B, gpus_per_server=2, max_batch_size=8,
            max_length=8192,
            pages_per_workload=100,
            workloads=workloads,
        ),
        # 4-GPU configs (2 servers) — more headroom
        MAXBenchmarkConfig(
            config_id="max_32b_4gpu_bs8",
            model=MODEL_32B, gpus_per_server=4, max_batch_size=8,
            max_length=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        MAXBenchmarkConfig(
            config_id="max_32b_4gpu_bs16",
            model=MODEL_32B, gpus_per_server=4, max_batch_size=16,
            max_length=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        # 8-GPU (1 server) — max memory, lowest latency
        MAXBenchmarkConfig(
            config_id="max_32b_8gpu_bs16",
            model=MODEL_32B, gpus_per_server=8, max_batch_size=16,
            max_length=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
    ]
    return configs


def get_all_configs() -> list[MAXBenchmarkConfig]:
    """Return all benchmark configurations."""
    return get_7b_configs() + get_32b_configs()


if __name__ == "__main__":
    print("7B Configurations (bfloat16):")
    for cfg in get_7b_configs():
        print(f"  {cfg.config_id}: gpus/srv={cfg.gpus_per_server}, "
              f"servers={cfg.num_servers}, batch={cfg.max_batch_size}, "
              f"cache={cfg.cache_strategy}")

    print("\n32B Configurations (bfloat16):")
    for cfg in get_32b_configs():
        print(f"  {cfg.config_id}: gpus/srv={cfg.gpus_per_server}, "
              f"servers={cfg.num_servers}, batch={cfg.max_batch_size}, "
              f"cache={cfg.cache_strategy}")

    print(f"\nTotal configs: {len(get_all_configs())}")
