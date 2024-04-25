import torch
import timeit
import logging
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.model import BasicsTransformerLM


# setup logging
logging.basicConfig(format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


@dataclass
class ModelArgs:
    vocab_size: int = 10000
    context_length: int = 128
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    attn_pdrop: Optional[float] = 0.1
    residual_pdrop: Optional[float] = 0.05


@dataclass
class TrainerArgs:
    batch_size: int = 16
    warmup_steps: int = 1
    train_steps: int = 5
    run_backward: bool = True


MODEL_CONFIGS = {
    "small": ModelArgs(
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
    ),
    "medium": ModelArgs(
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
    ),
    "large": ModelArgs(
        d_model=1280,
        num_layers=36,
        num_heads=20,
        d_ff=5120,
    ),
    "xl": ModelArgs(
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
    ),
    "2.7B": ModelArgs(
        d_model=2560,
        num_layers=32,
        num_heads=32,
        d_ff=10240,
    ),
}


def main(model_args: ModelArgs, trainer_args: TrainerArgs):
    model = BasicsTransformerLM(
        vocab_size=model_args.vocab_size,
        context_length=model_args.context_length,
        d_model=model_args.d_model,
        num_layers=model_args.num_layers,
        num_heads=model_args.num_heads,
        d_ff=model_args.d_ff,
        attn_pdrop=model_args.attn_pdrop,
        residual_pdrop=model_args.residual_pdrop,
    )
    model.to("cuda")
    model.train()
    dummy_data = torch.randint(
        0, model_args.vocab_size, (trainer_args.batch_size, model_args.context_length)
    ).to("cuda")
    dummy_labels = torch.randint(
        0, model_args.vocab_size, (trainer_args.batch_size, model_args.context_length)
    ).to("cuda")

    forward_times = []
    backward_times = []
    for i in range(trainer_args.warmup_steps + trainer_args.train_steps):
        start_time = timeit.default_timer()
        out = model(dummy_data)
        torch.cuda.synchronize()
        if i >= trainer_args.warmup_steps:
            forward_times.append(timeit.default_timer() - start_time)

        if not trainer_args.run_backward:
            continue

        loss = cross_entropy(out, dummy_labels)
        start_time = timeit.default_timer()
        loss.backward()
        torch.cuda.synchronize()
        if i >= trainer_args.warmup_steps:
            backward_times.append(timeit.default_timer() - start_time)
    
    logger.info(
        f"Forward: Avg = {np.mean(forward_times).round(4)}s, std: {np.std(forward_times).round(4)}s"
    )
    if trainer_args.run_backward:
        logger.info(
            f"Backward: Avg = {np.mean(backward_times).round(4)}s, std: {np.std(backward_times).round(4)}s"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-config",
        type=Literal["small", "medium", "large", "xl", "2.7B"],
        default="small",
    )
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=5)
    parser.add_argument("--run-backward", action="store_true", default=True)
    args = parser.parse_args()
    model_args = ModelArgs(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
    )
    trainer_args = TrainerArgs(
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        train_steps=args.train_steps,
        run_backward=args.run_backward,
    )
    main(model_args, trainer_args)
