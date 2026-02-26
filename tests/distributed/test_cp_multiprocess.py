"""
Multi-process gloo/CPU tests for context parallelism.

Uses torch.testing._internal.common_distributed.MultiProcessTestCase to spawn
real gloo processes, following the same pattern as PyTorch's FSDPTest.
"""

import sys

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA, run_tests

from fms.distributed.contextparallel import all_gather_from_context_parallel_region
from fms.utils.cp_wrapping import apply_gather_tensor_cp, apply_input_cp


class TestCPMultiProcess(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @classmethod
    def _run(cls, rank, test_name, file_name, pipe, **kwargs):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        dist.init_process_group(
            init_method=f"{FILE_SCHEMA}{file_name}",
            backend="gloo",
            world_size=self.world_size,
            rank=rank,
        )
        dist.barrier()
        self.run_test(test_name, pipe)

    def test_input_cp_roundtrip(self):
        """apply_input_cp + apply_gather_tensor_cp should recover the original tensor."""
        group = dist.group.WORLD
        torch.manual_seed(42)
        # b=2, seq=12, d=8 -- seq=12 divides evenly by world_size=2
        full_tensor = torch.randn(2, 12, 8)

        chunk, original_size = apply_input_cp(full_tensor, group, dim=1)
        self.assertEqual(chunk.size(1), 12 // self.world_size)
        self.assertEqual(original_size, 12)

        gathered = apply_gather_tensor_cp(
            chunk, group, dim=1, original_size=original_size
        )
        torch.testing.assert_close(gathered, full_tensor)

    def test_input_cp_roundtrip_padded(self):
        """Roundtrip works when seq length requires padding."""
        group = dist.group.WORLD
        torch.manual_seed(42)
        # b=2, seq=7, d=8 -- 7 is not divisible by 2
        full_tensor = torch.randn(2, 7, 8)

        chunk, original_size = apply_input_cp(full_tensor, group, dim=1)
        self.assertEqual(original_size, 7)

        gathered = apply_gather_tensor_cp(
            chunk, group, dim=1, original_size=original_size
        )
        self.assertEqual(gathered.size(1), 7)
        torch.testing.assert_close(gathered, full_tensor)

    def test_all_gather_context_parallel(self):
        """all_gather_from_context_parallel_region gathers along dim=0."""
        group = dist.group.WORLD
        rank = dist.get_rank()

        local = torch.full((2, 4), float(rank))
        gathered = all_gather_from_context_parallel_region(local, rank=rank, pg=group)
        self.assertEqual(gathered.shape, (2 * self.world_size, 4))

        for r in range(self.world_size):
            expected = torch.full((2, 4), float(r))
            torch.testing.assert_close(gathered[r * 2 : (r + 1) * 2], expected)

    def test_all_gather_backward_splits(self):
        """Backward pass of all_gather should split the gradient."""
        group = dist.group.WORLD
        rank = dist.get_rank()

        local = torch.randn(3, 4, requires_grad=True)
        gathered = all_gather_from_context_parallel_region(local, rank=rank, pg=group)
        self.assertEqual(gathered.shape, (3 * self.world_size, 4))

        loss = gathered.sum()
        loss.backward()

        self.assertIsNotNone(local.grad)
        self.assertEqual(local.grad.shape, (3, 4))
        torch.testing.assert_close(local.grad, torch.ones(3, 4))


if __name__ == "__main__":
    run_tests()
