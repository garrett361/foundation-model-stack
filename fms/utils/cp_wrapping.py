import torch
import torch.distributed._functional_collectives as funcol
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms.distributed import rank_and_world


def apply_layer_cp(model: nn.Module, group: ProcessGroup) -> nn.Module:
    """Recursively wrap modules that have a `to_cp` method."""
    if hasattr(model, "to_cp") and callable(model.to_cp):
        return model.to_cp(group)
    for name, layer in model.named_children():
        setattr(model, name, apply_layer_cp(layer, group))
    return model


def _pad_to_multiple(
    tensor: torch.Tensor, dim: int, multiple: int
) -> torch.Tensor:
    """Pad tensor with zeros along `dim` so its size is a multiple of `multiple`."""
    size = tensor.size(dim)
    remainder = size % multiple
    if remainder == 0:
        return tensor
    pad_amount = multiple - remainder
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_amount
    padding = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, padding], dim=dim)


def apply_input_cp(
    tensor: torch.Tensor, group: ProcessGroup, dim: int = 1
) -> tuple[torch.Tensor, int]:
    """Split `tensor` along `dim` across the CP group.

    If the tensor size along `dim` is not evenly divisible by the world size,
    the tensor is padded with zeros to the next multiple before splitting.

    Returns (chunk, original_size) so the caller can later trim padding after
    gathering.
    """
    rank, world_size = rank_and_world(group)
    original_size = tensor.size(dim)
    tensor = _pad_to_multiple(tensor, dim, world_size)
    chunk = tensor.tensor_split(world_size, dim=dim)[rank]
    return chunk, original_size


def apply_gather_tensor_cp(
    tensor: torch.Tensor,
    group: ProcessGroup,
    dim: int = 1,
    original_size: int | None = None,
) -> torch.Tensor:
    """Gather `tensor` along `dim` across the CP group.

    If `original_size` is provided, the gathered tensor is trimmed back to that
    size along `dim` to remove any padding introduced by `apply_input_cp`.
    """
    gathered = funcol.all_gather_tensor(
        tensor.contiguous(), gather_dim=dim, group=group
    )
    if original_size is not None:
        gathered = gathered.narrow(dim, 0, original_size)
    return gathered
