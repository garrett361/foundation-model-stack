"""
Multi-process gloo/CPU tests for context parallelism.

Uses torch.testing._internal.common_distributed.MultiProcessTestCase to spawn
real gloo processes, following the same pattern as PyTorch's FSDPTest.
"""

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import FILE_SCHEMA

from fms.distributed.contextparallel import all_gather_from_context_parallel_region
from fms.modules.attention import MultiHeadAttention
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

    def _make_attention(self, emb_dim=64, nheads=4, kvheads=2):
        return MultiHeadAttention(
            emb_dim=emb_dim,
            emb_kq=emb_dim // nheads,
            emb_v=emb_dim // nheads,
            nheads=nheads,
            kvheads=kvheads,
            fused=True,
        )

    def test_cp_forward_causal_matches_reference(self):
        """Gathered CP causal attention output should match single-process."""
        group = dist.group.WORLD
        rank = dist.get_rank()

        emb_dim, nheads, kvheads = 64, 4, 2
        seq_len, batch_size = 8, 2

        # Same seed on all ranks -> same weights and input
        torch.manual_seed(42)
        attention = self._make_attention(emb_dim, nheads, kvheads)
        attention.eval()
        x_full = torch.randn(batch_size, seq_len, emb_dim)

        # Reference: single-process on full sequence
        with torch.no_grad():
            ref_out = attention(x_full, is_causal_mask=True)

        # CP attention with same weights
        cp_attention = attention.to_cp(group)
        cp_attention.load_state_dict(attention.state_dict())
        cp_attention.eval()

        # Each rank gets its chunk of the sequence
        local_seq = seq_len // self.world_size
        x_local = x_full[:, rank * local_seq : (rank + 1) * local_seq, :]

        with torch.no_grad():
            cp_out = cp_attention(x_local, is_causal_mask=True)

        # Gather outputs across ranks and compare
        gathered = apply_gather_tensor_cp(cp_out, group, dim=1)
        torch.testing.assert_close(gathered, ref_out)

    def test_cp_forward_bidirectional_matches_reference(self):
        """Gathered CP bidirectional attention output should match single-process."""
        group = dist.group.WORLD
        rank = dist.get_rank()

        emb_dim, nheads, kvheads = 64, 4, 2
        seq_len, batch_size = 8, 2

        torch.manual_seed(42)
        attention = self._make_attention(emb_dim, nheads, kvheads)
        attention.eval()
        x_full = torch.randn(batch_size, seq_len, emb_dim)

        with torch.no_grad():
            ref_out = attention(x_full, is_causal_mask=False)

        cp_attention = attention.to_cp(group)
        cp_attention.load_state_dict(attention.state_dict())
        cp_attention.eval()

        local_seq = seq_len // self.world_size
        x_local = x_full[:, rank * local_seq : (rank + 1) * local_seq, :]

        with torch.no_grad():
            cp_out = cp_attention(x_local, is_causal_mask=False)

        gathered = apply_gather_tensor_cp(cp_out, group, dim=1)
        torch.testing.assert_close(gathered, ref_out)

    def test_cp_backward_matches_reference(self):
        """CP attention input and weight gradients should match single-process."""
        group = dist.group.WORLD
        rank = dist.get_rank()

        emb_dim, nheads, kvheads = 64, 4, 2
        seq_len, batch_size = 8, 2

        # Reference forward + backward
        torch.manual_seed(42)
        ref_attention = self._make_attention(emb_dim, nheads, kvheads)
        ref_attention.train()
        x_full = torch.randn(batch_size, seq_len, emb_dim, requires_grad=True)

        ref_out = ref_attention(x_full, is_causal_mask=True)
        ref_out.sum().backward()
        ref_input_grad = x_full.grad.clone()
        ref_weight_grads = {
            name: p.grad.clone() for name, p in ref_attention.named_parameters()
            if p.grad is not None
        }

        # CP forward + backward with same initial weights
        torch.manual_seed(42)
        cp_attention = self._make_attention(emb_dim, nheads, kvheads)
        cp_attention.load_state_dict(ref_attention.state_dict())
        cp_attention = cp_attention.to_cp(group)
        cp_attention.load_state_dict(ref_attention.state_dict())
        cp_attention.train()

        local_seq = seq_len // self.world_size
        x_local = (
            x_full.detach()[:, rank * local_seq : (rank + 1) * local_seq, :]
            .clone()
            .requires_grad_(True)
        )

        cp_out = cp_attention(x_local, is_causal_mask=True)
        cp_out.sum().backward()

        # Input grad: each rank's grad should match the corresponding slice
        ref_input_grad_local = ref_input_grad[
            :, rank * local_seq : (rank + 1) * local_seq, :
        ]
        torch.testing.assert_close(x_local.grad, ref_input_grad_local)

        # Weight grads: CP grads summed across ranks should match reference grads.
        # Each rank sees only a chunk of the sequence, so its local weight grad is
        # a partial sum. all-reduce to get the full gradient.
        for name, p in cp_attention.named_parameters():
            if p.grad is None:
                continue
            # Sum gradients across CP ranks
            grad_sum = p.grad.clone()
            dist.all_reduce(grad_sum, op=dist.ReduceOp.SUM)
            self.assertIn(name, ref_weight_grads, f"unexpected param {name}")
            torch.testing.assert_close(grad_sum, ref_weight_grads[name])
