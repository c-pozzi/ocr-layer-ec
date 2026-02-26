#!/usr/bin/env python3
"""
Continuous Batching Benchmark
Tests throughput scaling with different concurrency levels across 8 GPU servers.
"""

import asyncio
import aiohttp
import time
import argparse
import base64
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import json

# Server configuration
SERVERS = [f"http://localhost:{8000 + i}" for i in range(8)]

@dataclass
class RequestResult:
    file: str
    server: int
    start_time: float
    end_time: float
    input_tokens: int = 0
    output_tokens: int = 0
    success: bool = True
    error: str = ""
    
    @property
    def latency(self):
        return self.end_time - self.start_time

def encode_image_base64(path: Path) -> str:
    """Encode image/PDF to base64."""
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")

def get_files(folder: Path, extensions: List[str] = [".pdf", ".png", ".tif", ".tiff"]) -> List[Path]:
    """Get all processable files from folder."""
    files = []
    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))
    return sorted(files)

async def send_request(
    session: aiohttp.ClientSession,
    server_url: str,
    server_idx: int,
    file_path: Path,
    prompt: str,
    max_tokens: int
) -> RequestResult:
    """Send a single OCR request."""
    
    start_time = time.time()
    
    try:
        # Encode file
        file_data = encode_image_base64(file_path)
        suffix = file_path.suffix.lower()
        
        if suffix == ".pdf":
            media_type = "application/pdf"
        elif suffix in [".tif", ".tiff"]:
            media_type = "image/tiff"
        elif suffix == ".png":
            media_type = "image/png"
        else:
            media_type = "image/jpeg"
        
        payload = {
            "model": "Qwen/Qwen2.5-VL-7B-Instruct",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{media_type};base64,{file_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }],
            "max_tokens": max_tokens,
            "temperature": 0
        }
        
        async with session.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=300)
        ) as response:
            result = await response.json()
        
        end_time = time.time()
        
        if "error" in result:
            return RequestResult(
                file=file_path.name,
                server=server_idx,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error=str(result["error"])
            )
        
        usage = result.get("usage", {})
        
        return RequestResult(
            file=file_path.name,
            server=server_idx,
            start_time=start_time,
            end_time=end_time,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            success=True
        )
        
    except Exception as e:
        return RequestResult(
            file=file_path.name,
            server=server_idx,
            start_time=start_time,
            end_time=time.time(),
            success=False,
            error=str(e)
        )

async def run_benchmark(
    files: List[Path],
    concurrency_per_server: int,
    prompt: str,
    max_tokens: int
) -> List[RequestResult]:
    """Run benchmark distributing files across 8 servers."""
    
    results = []
    total_concurrency = concurrency_per_server * len(SERVERS)
    semaphore = asyncio.Semaphore(total_concurrency)
    
    # Track which server to use (round-robin)
    server_idx = 0
    
    async def process_file(file_path: Path, assigned_server: int):
        async with semaphore:
            connector = aiohttp.TCPConnector(limit=100)
            async with aiohttp.ClientSession(connector=connector) as session:
                result = await send_request(
                    session,
                    SERVERS[assigned_server],
                    assigned_server,
                    file_path,
                    prompt,
                    max_tokens
                )
                results.append(result)
                
                # Progress indicator
                done = len(results)
                total = len(files)
                if done % 10 == 0 or done == total:
                    print(f"  Progress: {done}/{total} ({done/total*100:.0f}%)")
                
                return result
    
    # Assign files to servers round-robin
    tasks = []
    for i, file_path in enumerate(files):
        server = i % len(SERVERS)
        tasks.append(process_file(file_path, server))
    
    await asyncio.gather(*tasks)
    
    return results

def analyze_results(results: List[RequestResult], concurrency_per_server: int):
    """Analyze and print benchmark results."""
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    if not successful:
        print("  ❌ All requests failed!")
        for r in failed[:5]:
            print(f"    {r.file}: {r.error}")
        return None
    
    total_time = max(r.end_time for r in successful) - min(r.start_time for r in successful)
    total_output_tokens = sum(r.output_tokens for r in successful)
    total_input_tokens = sum(r.input_tokens for r in successful)
    
    latencies = [r.latency for r in successful]
    avg_latency = sum(latencies) / len(latencies)
    latencies_sorted = sorted(latencies)
    p50 = latencies_sorted[len(latencies_sorted) // 2]
    p95 = latencies_sorted[int(len(latencies_sorted) * 0.95)]
    p99 = latencies_sorted[int(len(latencies_sorted) * 0.99)]
    
    # Per-server stats
    server_counts = {}
    for r in successful:
        server_counts[r.server] = server_counts.get(r.server, 0) + 1
    
    stats = {
        "concurrency_per_server": concurrency_per_server,
        "total_concurrency": concurrency_per_server * 8,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "total_time_sec": total_time,
        "requests_per_sec": len(successful) / total_time,
        "output_tokens_per_sec": total_output_tokens / total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "avg_latency": avg_latency,
        "p50_latency": p50,
        "p95_latency": p95,
        "p99_latency": p99,
    }
    
    print(f"\n{'='*60}")
    print(f"Concurrency: {concurrency_per_server}/server × 8 = {concurrency_per_server * 8} total")
    print(f"{'='*60}")
    print(f"Requests:       {len(successful)}/{len(results)} successful")
    print(f"Total time:     {total_time:.1f}s")
    print(f"")
    print(f"Throughput:")
    print(f"  Requests/sec:     {stats['requests_per_sec']:.2f}")
    print(f"  Tokens out/sec:   {stats['output_tokens_per_sec']:.1f}")
    print(f"")
    print(f"Latency:")
    print(f"  Average:  {avg_latency:.2f}s")
    print(f"  P50:      {p50:.2f}s")
    print(f"  P95:      {p95:.2f}s")
    print(f"  P99:      {p99:.2f}s")
    print(f"")
    print(f"Server distribution: {dict(sorted(server_counts.items()))}")
    
    if failed:
        print(f"\n⚠️ {len(failed)} failed requests")
    
    return stats

async def main():
    parser = argparse.ArgumentParser(description="Continuous Batching Benchmark")
    parser.add_argument("folder", help="Folder containing PDF/image files")
    parser.add_argument("--concurrency", "-c", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Concurrency levels per server to test (default: 1 2 4 8)")
    parser.add_argument("--max-files", "-n", type=int, default=None,
                        help="Max files to process (default: all)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max output tokens (default: 2048)")
    parser.add_argument("--prompt", default="Extract all text from this document.",
                        help="Prompt to use")
    
    args = parser.parse_args()
    
    # Get files
    folder = Path(args.folder)
    files = get_files(folder)
    
    if args.max_files:
        files = files[:args.max_files]
    
    print(f"Found {len(files)} files in {folder}")
    print(f"Testing concurrency levels: {args.concurrency}")
    print(f"Servers: {SERVERS}")
    print("")
    
    # Check servers are up
    print("Checking servers...")
    async with aiohttp.ClientSession() as session:
        for i, url in enumerate(SERVERS):
            try:
                async with session.get(f"{url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        print(f"  ✅ Server {i} ({url}) ready")
                    else:
                        print(f"  ❌ Server {i} ({url}) returned {resp.status}")
            except Exception as e:
                print(f"  ❌ Server {i} ({url}) not reachable: {e}")
                return
    print("")
    
    # Run benchmarks
    all_stats = []
    
    for conc in args.concurrency:
        print(f"\n>>> Testing concurrency={conc} per server ({conc * 8} total)...")
        
        results = await run_benchmark(
            files,
            conc,
            args.prompt,
            args.max_tokens
        )
        
        stats = analyze_results(results, conc)
        if stats:
            all_stats.append(stats)
    
    # Summary table
    if len(all_stats) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY: Continuous Batching Effect")
        print(f"{'='*70}")
        print(f"{'Conc/srv':<10} {'Total':<8} {'Req/s':<10} {'Tok/s':<12} {'Avg Lat':<10} {'P95 Lat':<10}")
        print("-" * 70)
        for s in all_stats:
            print(f"{s['concurrency_per_server']:<10} {s['total_concurrency']:<8} {s['requests_per_sec']:<10.2f} {s['output_tokens_per_sec']:<12.1f} {s['avg_latency']:<10.2f} {s['p95_latency']:<10.2f}")
        
        # Show speedup
        if len(all_stats) >= 2:
            baseline = all_stats[0]['requests_per_sec']
            best = max(s['requests_per_sec'] for s in all_stats)
            print(f"\n🚀 Throughput improvement: {best/baseline:.1f}x (from concurrency={all_stats[0]['concurrency_per_server']} to best)")
    
    # Save results
    output_file = Path(__file__).resolve().parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(main())