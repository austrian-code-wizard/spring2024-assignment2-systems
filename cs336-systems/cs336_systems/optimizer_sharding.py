import torch
from typing import Any, Type
import torch.distributed as dist


class OptimizerSharded(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[torch.optim.Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizers = {}
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        super().__init__(params, defaults={})

    def step(self, closure=None):
        handlers = []
        for group in self.param_groups:
            for p in group["params"]:
                optimizer = self.optimizers[p]
                if not isinstance(optimizer, int):
                    optimizer.step(closure)
                handlers.append(dist.broadcast(p.data, optimizer if isinstance(optimizer, int) else self.rank, async_op=True))
        for handle in handlers:
            handle.wait()

    def add_param_group(self, param_group: dict[str, Any]):
        for param in param_group["params"]:
            target_rank = len(self.optimizers) % self.world_size
            if param.requires_grad and target_rank == self.rank:
                self.optimizers[param] = self.optimizer_cls([param], **self.kwargs)
            else:
                self.optimizers[param] = target_rank
        super().add_param_group(param_group)
