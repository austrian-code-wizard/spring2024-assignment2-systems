from typing import Any
import torch
import torch.distributed as dist
from dataclasses import dataclass


@dataclass
class Bucket:
    params: list[torch.Tensor]
    num_total: int
    num_ready: int
    size: float
    flattened_grads: torch.Tensor


class DDPBucketed:
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        self.module = module
        self.handles = []

        size_in_mb = (
            lambda tensor: tensor.element_size() * tensor.nelement() / (1024 * 1024)
        )

        param_sizes = [
            size_in_mb(p.data) if p.requires_grad else None
            for p in self.module.parameters()
        ]

        self.buckets: list[Bucket] = []
        self.params_2_buckets: dict[torch.Tensor, Bucket] = {}
        for param, size in zip(
            reversed(list(self.module.parameters())), reversed(param_sizes)
        ):
            if size is None:
                continue
            if len(self.buckets) == 0 or (
                self.buckets[-1].size + size > bucket_size_mb
                and self.buckets[-1].size > 0
            ):
                self.buckets.append(Bucket([], 0, 0, 0.0, None))
            self.buckets[-1].params.append(param)
            self.buckets[-1].num_total += 1
            self.params_2_buckets[param] = self.buckets[-1]
            self.buckets[-1].size += size

        self.sync_weights()
        self.world_size = dist.get_world_size()
        self.backend = dist.get_backend()
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(self.sync_gradient)

    def sync_weights(self):
        for param in self.module.parameters():
            self.handles.append(dist.broadcast(param.data, 0, async_op=True))
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

    def sync_gradient(self, p: torch.Tensor):
        b = self.params_2_buckets[p]
        b.num_ready += 1
        if b.num_ready == b.num_total:
            grads = [param.grad for param in b.params]
            b.flattened_grads = torch._utils._flatten_dense_tensors(grads)
            if self.backend == "nccl":
                self.handles.append(
                    dist.all_reduce(
                        tensor=b.flattened_grads, op=dist.ReduceOp.AVG, async_op=True
                    )
                )
            else:
                self.handles.append(
                    dist.all_reduce(
                        tensor=b.flattened_grads, op=dist.ReduceOp.SUM, async_op=True
                    )
                )
            self.params_2_buckets[p].num_ready = 0

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()

        for b in self.buckets:
            for param, grad in zip(
                b.params,
                torch._utils._unflatten_dense_tensors(
                    b.flattened_grads, [p.grad for p in b.params]
                ),
            ):
                param.grad = grad

        if self.backend == "gloo":
            for p in self.module.parameters():
                if p.requires_grad:
                    p.grad /= self.world_size

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.module(*args, **kwds)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
