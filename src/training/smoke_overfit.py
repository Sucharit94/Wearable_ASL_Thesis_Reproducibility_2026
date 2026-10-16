from __future__ import annotations

import math
import random
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch import nn

from src.data.collate import collate_numpy
from src.training.ctc_loss import torch_ctc_loss_from_logits


def set_deterministic_seed(seed: int) -> None:
    seed = int(seed)

    if seed < 0:
        raise ValueError("seed must be non-negative.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def batch_slices(
    items: Sequence[dict],
    batch_size: int,
) -> Iterator[Sequence[dict]]:
    batch_size = int(batch_size)

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _spec_value(specification: Any, name: str) -> Any:
    if isinstance(specification, Mapping):
        return specification[name]

    return getattr(specification, name)


def build_optimizer(
    model: nn.Module,
    specification: Any,
) -> torch.optim.Optimizer:
    optimizer_name = str(_spec_value(specification, "optimizer"))
    learning_rate = float(_spec_value(specification, "learning_rate"))
    weight_decay = float(_spec_value(specification, "weight_decay"))

    keyword_arguments = {
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    if optimizer_name == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            **keyword_arguments,
        )

    if optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            **keyword_arguments,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}.")


def is_out_of_memory_error(error: BaseException) -> bool:
    message = str(error).lower()

    markers = (
        "out of memory",
        "cannot allocate memory",
        "not enough memory",
        "defaultcpuallocator",
    )

    return any(marker in message for marker in markers)


def _batch_to_torch(
    items: Sequence[dict],
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    batch = collate_numpy(items)

    inputs = torch.as_tensor(
        batch["inputs"],
        dtype=torch.float32,
        device=device,
    )
    input_lengths = torch.as_tensor(
        batch["input_lengths"],
        dtype=torch.long,
        device="cpu",
    )
    targets_padded = torch.as_tensor(
        batch["targets_padded"],
        dtype=torch.long,
        device="cpu",
    )
    target_lengths = torch.as_tensor(
        batch["target_lengths"],
        dtype=torch.long,
        device="cpu",
    )

    return (
        inputs,
        input_lengths,
        targets_padded,
        target_lengths,
    )


def _ctc_loss(
    model: nn.Module,
    items: Sequence[dict],
    device: torch.device,
) -> tuple[torch.Tensor, object]:
    (
        inputs,
        input_lengths,
        targets_padded,
        target_lengths,
    ) = _batch_to_torch(items, device)

    output = model(
        inputs,
        input_lengths,
    )

    loss = torch_ctc_loss_from_logits(
        output.logits_btc,
        targets_padded,
        output.output_lengths,
        target_lengths,
        reduction="mean",
    )

    return loss, output


def one_batch_smoke(
    model: nn.Module,
    items: Sequence[dict],
    *,
    device: torch.device,
) -> dict:
    if not items:
        raise ValueError("Smoke batch cannot be empty.")

    model.to(device)
    model.train()
    model.zero_grad(set_to_none=True)

    loss, output = _ctc_loss(
        model,
        items,
        device,
    )

    if not torch.isfinite(loss).item():
        raise RuntimeError("Smoke loss is not finite.")

    loss.backward()

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    missing = [
        name
        for name, parameter in trainable
        if parameter.grad is None
    ]

    nonfinite = [
        name
        for name, parameter in trainable
        if parameter.grad is not None
        and not torch.isfinite(parameter.grad).all().item()
    ]

    gradient_tensors = sum(
        parameter.grad is not None
        for _, parameter in trainable
    )

    squared_norm = 0.0

    for _, parameter in trainable:
        if parameter.grad is not None:
            squared_norm += float(
                parameter.grad
                .detach()
                .double()
                .pow(2)
                .sum()
                .item()
            )

    result = {
        "batch_size": len(items),
        "loss": float(
            loss.detach().cpu().item()
        ),
        "loss_finite": True,
        "logit_shape": [
            int(value)
            for value in output.logits_btc.shape
        ],
        "output_lengths": [
            int(value)
            for value in torch.as_tensor(
                output.output_lengths
            ).tolist()
        ],
        "gradient_tensors": int(gradient_tensors),
        "missing_gradients": len(missing),
        "nonfinite_gradients": len(nonfinite),
        "gradient_global_norm": math.sqrt(
            squared_norm
        ),
        "passed": (
            not missing
            and not nonfinite
            and gradient_tensors > 0
            and squared_norm > 0.0
        ),
    }

    model.zero_grad(set_to_none=True)

    if not result["passed"]:
        raise RuntimeError(
            "One-batch smoke test failed: "
            f"missing={missing}, "
            f"nonfinite={nonfinite}."
        )

    return result


@torch.no_grad()
def evaluate_average_loss(
    model: nn.Module,
    items: Sequence[dict],
    *,
    batch_size: int,
    device: torch.device,
) -> float:
    if not items:
        raise ValueError(
            "Evaluation items cannot be empty."
        )

    model.eval()
    weighted_loss = 0.0
    example_count = 0

    for item_batch in batch_slices(
        items,
        batch_size,
    ):
        loss, _ = _ctc_loss(
            model,
            item_batch,
            device,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                "Tiny-overfit evaluation loss "
                "is not finite."
            )

        count = len(item_batch)

        weighted_loss += (
            float(loss.cpu().item())
            * count
        )
        example_count += count

    return weighted_loss / example_count


def tiny_overfit(
    model: nn.Module,
    items: Sequence[dict],
    specification: Any,
    *,
    batch_size: int,
    minimum_epochs: int,
    maximum_epochs: int,
    minimum_relative_decrease: float,
    device: torch.device,
) -> dict:
    if not items:
        raise ValueError(
            "Tiny-overfit items cannot be empty."
        )

    if not (
        1
        <= minimum_epochs
        <= maximum_epochs
    ):
        raise ValueError(
            "Invalid tiny-overfit epoch bounds."
        )

    if minimum_relative_decrease <= 0.0:
        raise ValueError(
            "minimum_relative_decrease "
            "must be positive."
        )

    model.to(device)

    optimizer = build_optimizer(
        model,
        specification,
    )

    gradient_clip_norm = float(
        _spec_value(
            specification,
            "gradient_clip_norm",
        )
    )

    initial_loss = evaluate_average_loss(
        model,
        items,
        batch_size=batch_size,
        device=device,
    )

    best_loss = initial_loss

    history = [
        {
            "epoch": 0,
            "training_loss": None,
            "evaluation_loss": initial_loss,
            "relative_decrease_from_initial": 0.0,
            "maximum_preclip_gradient_norm": None,
        }
    ]

    epochs_completed = 0

    for epoch in range(
        1,
        maximum_epochs + 1,
    ):
        model.train()

        weighted_training_loss = 0.0
        training_examples = 0
        maximum_gradient_norm = 0.0

        for item_batch in batch_slices(
            items,
            batch_size,
        ):
            optimizer.zero_grad(
                set_to_none=True
            )

            loss, _ = _ctc_loss(
                model,
                item_batch,
                device,
            )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Tiny-overfit training loss "
                    "is not finite."
                )

            loss.backward()

            gradient_norm = (
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                    error_if_nonfinite=True,
                )
            )

            optimizer.step()

            count = len(item_batch)

            weighted_training_loss += (
                float(
                    loss.detach().cpu().item()
                )
                * count
            )
            training_examples += count

            maximum_gradient_norm = max(
                maximum_gradient_norm,
                float(
                    torch.as_tensor(
                        gradient_norm
                    )
                    .cpu()
                    .item()
                ),
            )

        evaluation_loss = (
            evaluate_average_loss(
                model,
                items,
                batch_size=batch_size,
                device=device,
            )
        )

        best_loss = min(
            best_loss,
            evaluation_loss,
        )

        epochs_completed = epoch

        relative_decrease = (
            initial_loss - best_loss
        ) / initial_loss

        history.append(
            {
                "epoch": epoch,
                "training_loss": (
                    weighted_training_loss
                    / training_examples
                ),
                "evaluation_loss": (
                    evaluation_loss
                ),
                "relative_decrease_from_initial": (
                    relative_decrease
                ),
                "maximum_preclip_gradient_norm": (
                    maximum_gradient_norm
                ),
            }
        )

        if (
            epoch >= minimum_epochs
            and relative_decrease
            >= minimum_relative_decrease
        ):
            break

    final_loss = float(
        history[-1]["evaluation_loss"]
    )

    relative_decrease = (
        initial_loss - best_loss
    ) / initial_loss

    passed = (
        math.isfinite(initial_loss)
        and math.isfinite(best_loss)
        and best_loss < initial_loss
        and relative_decrease
        >= minimum_relative_decrease
    )

    return {
        "initial_evaluation_loss": initial_loss,
        "final_evaluation_loss": final_loss,
        "best_evaluation_loss": best_loss,
        "absolute_decrease": (
            initial_loss - best_loss
        ),
        "relative_decrease": relative_decrease,
        "minimum_relative_decrease_required": (
            minimum_relative_decrease
        ),
        "minimum_epochs": minimum_epochs,
        "maximum_epochs": maximum_epochs,
        "epochs_completed": epochs_completed,
        "history": history,
        "passed": passed,
    }
