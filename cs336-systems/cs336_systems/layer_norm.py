from cs336_basics.model import RMSNorm
from cs336_systems.rms import RMSNormTriton
import torch
import time


ITERS = 1000
ROWS = 50000
COLS = [1024, 2048, 4096, 8192]


def main():
    for col in COLS:
        data = torch.randn(ROWS, col).to("cuda")

        layer_norm = torch.nn.LayerNorm(col).to("cuda")
        rms_norm = RMSNorm(col).to("cuda")

        # warmup
        for _ in range(10):
            layer_norm(data)
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERS):
            layer_norm(data)
            torch.cuda.synchronize()
        end = time.time()
        print(f"LayerNorm: {col} cols took {end - start} seconds")

        # warmup
        for _ in range(10):
            rms_norm(data)
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERS):
            rms_norm(data)
            torch.cuda.synchronize()
        end = time.time()
        print(f"RMSNorm: {col} cols took {end - start} seconds")

        # warmup
        weight = torch.randn(col).to("cuda")
        for _ in range(10):
            RMSNormTriton.apply(data, weight)
            torch.cuda.synchronize()
        start = time.time()
        for _ in range(ITERS):
            RMSNormTriton.apply(data, weight)
            torch.cuda.synchronize()
        end = time.time()
        print(f"RMSNormTriton: {col} cols took {end - start} seconds")


if __name__ == "__main__":
    main()