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
        result = torch.randn(ROWS, col).to("cuda")
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
                output.backward(result)
                torch.cuda.synchronize()
                backward.append(time.time() - start)
                data.grad = None
        print(f"LayerNorm: {col} cols took {sum(forward)} seconds")
        if run_backward:
            print(f"LayerNorm backward: {col} cols took {sum(backward)} seconds")

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
                output.backward(result)
                torch.cuda.synchronize()
                backward_rms.append(time.time() - start_rms)
                data.grad = None
        print(f"RMSNorm: {col} cols took {sum(forward_rms)} seconds")
        if run_backward:
            print(f"RMSNorm backward: {col} cols took {sum(backward_rms)} seconds")

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
            #torch.cuda.synchronize()
            forward_triton.append(time.time() - start_triton)
            if run_backward:
                start_triton = time.time()
                output.backward(result)
                torch.cuda.synchronize()
                backward_triton.append(time.time() - start_triton)
                data.grad = None
        print(f"RMSNormTriton: {col} cols took {sum(forward_triton)} seconds")
        if run_backward:
            print(f"RMSNormTriton backward: {col} cols took {sum(backward_triton)} seconds")


if __name__ == "__main__":
    main(run_backward=True)