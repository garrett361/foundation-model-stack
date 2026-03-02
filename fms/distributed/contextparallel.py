# mypy: disable-error-code="method-assign,misc"

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol


def _all_gather(input_: torch.Tensor, pg: dist.ProcessGroup) -> torch.Tensor:
    """Gather the input tensor across the context parallel group."""
    if pg.size() == 1:
        return input_
    return funcol.all_gather_tensor(input_.contiguous(), gather_dim=0, group=pg)


def _split(
    input_: torch.Tensor, rank: int, pg: dist.ProcessGroup
) -> torch.Tensor:
    """Split the tensor along dim 0 and keep the corresponding slice."""
    if pg.size() == 1:
        return input_
    chunk_size = input_.size(0) // pg.size()
    input_list = torch.split(input_, chunk_size, dim=0)
    return input_list[rank].contiguous()


class _AllGatherFromContextParallelRegion(torch.autograd.Function):
    """Gather the input from the context parallel region (split on backward)."""

    @staticmethod
    def symbolic(
        graph, input_: torch.Tensor, rank: int, pg: dist.ProcessGroup
    ) -> torch.Tensor:
        return _all_gather(input_, pg)

    @staticmethod
    def forward(
        ctx, input_: torch.Tensor, rank: int, pg: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.rank = rank
        ctx.pg = pg
        return _all_gather(input_, pg)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return _split(grad_output, ctx.rank, ctx.pg), None, None


def all_gather_from_context_parallel_region(
    input_: torch.Tensor, rank: int, pg: dist.ProcessGroup
) -> torch.Tensor:
    return _AllGatherFromContextParallelRegion.apply(input_, rank, pg)
