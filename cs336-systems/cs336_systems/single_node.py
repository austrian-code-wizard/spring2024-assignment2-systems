import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import logging
import argparse
from typing import Dict

# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

TENSOR_SIZES: Dict[str, int] = {
    "512KB": int(512e3 / 4),
    "1MB": int(1e6 / 4),
    "10MB": int(10e6 / 4),
    "50MB": int(50e6 / 4),
    "100MB": int(100e6 / 4),
    "500MB": int(500e6 / 4),
    "1GB": int(1e9 / 4),
}


def setup(rank: int, world_size: int, backend: str, use_cuda: bool) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if backend == "nccl" or (backend == "gloo" and use_cuda):
        torch.cuda.set_device(rank)


def benchmark_all_reduce(
    rank: int, world_size: int, backend: str, tensor_size_str: str, use_cuda: bool
) -> None:
    setup(rank, world_size, backend, use_cuda)
    tensor_size = TENSOR_SIZES[tensor_size_str]
    device = torch.device(
        "cuda" if (backend == "nccl" or (backend == "gloo" and use_cuda)) else "cpu"
    )

    # Warmup steps
    data = torch.randn(tensor_size, device=device).to(torch.float32)
    for _ in range(5):
        dist.all_reduce(data, async_op=False)
        if device.type == "cuda":
            torch.cuda.synchronize()

    start_time = time.time()
    dist.all_reduce(data, async_op=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.time() - start_time

    all_times = [None] * world_size
    dist.all_gather_object(all_times, total_time)
    if rank == 0:
        avg_time = sum(all_times) / world_size
        logger.debug(
            f"Backend: {backend}, Device: {device}, World size: {world_size}, Tensor size: {tensor_size_str}, Average Time taken: {avg_time} seconds"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--tensor_size", type=str, default="all")
    parser.add_argument("--backend", type=str, default="all")
    parser.add_argument("--world_size", type=int, default=0)
    args = parser.parse_args()

    backends = ["gloo", "nccl"] if args.backend == "all" else [args.backend]
    world_sizes = [2, 4, 6] if args.world_size == 0 else [args.world_size]
    tensor_sizes = (
        TENSOR_SIZES.keys() if args.tensor_size == "all" else [args.tensor_size]
    )

    assert not (
        not args.use_cuda and "nccl" in backends
    ), "NCCL backend requires CUDA support."

    for backend in backends:
        for world_size in world_sizes:
            for tensor_size_str in tensor_sizes:
                if backend == "nccl" and not torch.cuda.is_available():
                    logger.info("Skipping NCCL backend since CUDA is not available.")
                    continue
                if backend == "nccl":
                    torch.cuda.empty_cache()
                mp.spawn(
                    fn=benchmark_all_reduce,
                    args=(world_size, backend, tensor_size_str, args.use_cuda),
                    nprocs=world_size,
                    join=True,
                )


if __name__ == "__main__":
    main()
