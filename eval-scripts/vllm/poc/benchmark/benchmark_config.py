"""
Benchmark Configuration Matrix

Defines all vLLM server configurations to test on A100 GPUs.
Each config specifies model, tensor parallelism, CUDA graphs, memory, and batching params.
"""

from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results" / "benchmark_a100"

# Model IDs
MODEL_7B = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
MODEL_32B = "Qwen/Qwen2.5-VL-32B-Instruct-AWQ"

# Cost per hour for p4d.24xlarge (on-demand)
P4D_COST_PER_HOUR = 32.77


@dataclass
class BenchmarkConfig:
    config_id: str
    model: str
    tp: int  # tensor parallelism
    cuda_graphs: bool  # False = --enforce-eager
    gpu_mem_util: float
    max_num_seqs: int
    max_model_len: int = 8192
    max_num_batched_tokens: int | None = None  # None = vLLM default
    quantization: str = "awq"
    dtype: str = "float16"

    # Benchmark parameters
    workloads: list[str] = field(default_factory=list)
    pages_per_workload: int = 200
    num_runs: int = 3
    concurrency_per_server: int = 8
    warmup_requests: int = 10

    @property
    def num_servers(self) -> int:
        return 8 // self.tp

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / self.config_id

    def vllm_serve_args(self, port: int) -> list[str]:
        """Generate vllm serve CLI arguments for one server instance."""
        args = [
            "vllm", "serve", self.model,
            "--port", str(port),
            "--quantization", self.quantization,
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_mem_util),
            "--max-num-seqs", str(self.max_num_seqs),
        ]
        if self.max_num_batched_tokens is not None:
            args.extend(["--max-num-batched-tokens", str(self.max_num_batched_tokens)])
        if not self.cuda_graphs:
            args.append("--enforce-eager")
        if self.tp > 1:
            args.extend(["--tensor-parallel-size", str(self.tp)])
        return args

    def cuda_devices_for_server(self, server_index: int) -> str:
        """Return CUDA_VISIBLE_DEVICES string for a given server index."""
        start = server_index * self.tp
        devices = list(range(start, start + self.tp))
        return ",".join(str(d) for d in devices)

    def server_urls(self) -> list[str]:
        """Return list of server URLs."""
        return [f"http://localhost:{8000 + i}" for i in range(self.num_servers)]


# =============================================================================
# 7B AWQ CONFIGS
# =============================================================================

def get_7b_configs() -> list[BenchmarkConfig]:
    """Return all 7B AWQ benchmark configurations."""
    workloads = ["classify", "simple_ocr"]

    configs = [
        BenchmarkConfig(
            config_id="7b_tp1_eager_m90",
            model=MODEL_7B, tp=1, cuda_graphs=False,
            gpu_mem_util=0.90, max_num_seqs=64,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="7b_tp1_graphs_m90",
            model=MODEL_7B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.90, max_num_seqs=64,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="7b_tp1_graphs_m95",
            model=MODEL_7B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=64,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="7b_tp1_graphs_m95_s128",
            model=MODEL_7B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=128,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="7b_tp1_graphs_m95_s256",
            model=MODEL_7B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=256,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="7b_tp1_graphs_m95_bt16k",
            model=MODEL_7B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=128,
            max_num_batched_tokens=16384,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="7b_tp1_graphs_m95_bt32k",
            model=MODEL_7B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=128,
            max_num_batched_tokens=32768,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="7b_tp2_graphs_m95",
            model=MODEL_7B, tp=2, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=128,
            workloads=workloads,
        ),
    ]
    return configs


# =============================================================================
# 32B AWQ CONFIGS
# =============================================================================

def get_32b_configs() -> list[BenchmarkConfig]:
    """Return all 32B AWQ benchmark configurations."""
    workloads = ["complex_ocr"]

    configs = [
        BenchmarkConfig(
            config_id="32b_tp1_graphs_m90",
            model=MODEL_32B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.90, max_num_seqs=32,
            max_model_len=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="32b_tp1_graphs_m95",
            model=MODEL_32B, tp=1, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=64,
            max_model_len=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="32b_tp2_eager_m95",
            model=MODEL_32B, tp=2, cuda_graphs=False,
            gpu_mem_util=0.95, max_num_seqs=64,
            max_model_len=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="32b_tp2_graphs_m95",
            model=MODEL_32B, tp=2, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=64,
            max_model_len=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="32b_tp2_graphs_m95_s128",
            model=MODEL_32B, tp=2, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=128,
            max_model_len=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="32b_tp4_graphs_m95",
            model=MODEL_32B, tp=4, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=64,
            max_model_len=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
        BenchmarkConfig(
            config_id="32b_tp8_graphs_m95",
            model=MODEL_32B, tp=8, cuda_graphs=True,
            gpu_mem_util=0.95, max_num_seqs=64,
            max_model_len=16384,
            pages_per_workload=100,
            workloads=workloads,
        ),
    ]
    return configs


def get_all_configs() -> list[BenchmarkConfig]:
    """Return all benchmark configurations."""
    return get_7b_configs() + get_32b_configs()


if __name__ == "__main__":
    print("7B AWQ Configurations:")
    for cfg in get_7b_configs():
        print(f"  {cfg.config_id}: TP={cfg.tp}, servers={cfg.num_servers}, "
              f"graphs={'on' if cfg.cuda_graphs else 'off'}, "
              f"mem={cfg.gpu_mem_util}, seqs={cfg.max_num_seqs}")

    print("\n32B AWQ Configurations:")
    for cfg in get_32b_configs():
        print(f"  {cfg.config_id}: TP={cfg.tp}, servers={cfg.num_servers}, "
              f"graphs={'on' if cfg.cuda_graphs else 'off'}, "
              f"mem={cfg.gpu_mem_util}, seqs={cfg.max_num_seqs}")

    print(f"\nTotal configs: {len(get_all_configs())}")
