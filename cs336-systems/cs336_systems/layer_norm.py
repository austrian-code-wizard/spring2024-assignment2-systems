from cs336_basics.model import RMSNorm
from cs336_systems.rms import RMSNormTriton
import torch
import time


ITERS = 1000
ROWS = 50000
COLS = [1024, 2048, 4096, 8192]


def main(run_backward: bool = False):
    for col in COLS:
        data = torch.randn(ROWS, col).to("cuda")

        layer_norm = torch.nn.LayerNorm(col).to("cuda")
        rms_norm = RMSNorm(col).to("cuda")

        # warmup
        for _ in range(10):
            layer_norm(data)
            torch.cuda.synchronize()
        forward = []
        backward = []
        for _ in range(ITERS):
            start = time.time()
            output = layer_norm(data)
            torch.cuda.synchronize()
            forward.append(time.time() - start)
            if run_backward:
                start = time.time()
                output.sum().backward()
                torch.cuda.synchronize()
                backward.append(time.time() - start)
        print(f"LayerNorm: {col} cols took {sum(forward) / ITERS} seconds")
        if run_backward:
            print(f"LayerNorm backward: {col} cols took {sum(backward) / ITERS} seconds")

        # warmup
        for _ in range(10):
            rms_norm(data)
            torch.cuda.synchronize()
        forward_rms = []
        backward_rms = []
        for _ in range(ITERS):
            start_rms = time.time()
            output = rms_norm(data)
            torch.cuda.synchronize()
            forward_rms.append(time.time() - start_rms)
            if run_backward:
                start_rms = time.time()
                output.sum().backward()
                torch.cuda.synchronize()
                backward_rms.append(time.time() - start_rms)
        print(f"RMSNorm: {col} cols took {sum(forward_rms) / ITERS} seconds")
        if run_backward:
            print(f"RMSNorm backward: {col} cols took {sum(backward_rms) / ITERS} seconds")

        triton_norm = RMSNormTriton(col).to("cuda")
        # warmup
        for _ in range(10):
            triton_norm(data)
            torch.cuda.synchronize()
        forward_triton = []
        backward_triton = []
        for _ in range(ITERS):
            start_triton = time.time()
            output = triton_norm(data)
            torch.cuda.synchronize()
            forward_triton.append(time.time() - start_triton)
            if run_backward:
                start_triton = time.time()
                output.sum().backward()
                torch.cuda.synchronize()
                backward_triton.append(time.time() - start_triton)
        print(f"RMSNormTriton: {col} cols took {sum(forward_triton) / ITERS} seconds")
        if run_backward:
            print(f"RMSNormTriton backward: {col} cols took {sum(backward_triton) / ITERS} seconds")


if __name__ == "__main__":
    main(run_backward=True)