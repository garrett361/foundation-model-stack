import pytest
import torch
import torch.distributed

from fms.modules.attention import CPMultiHeadAttention, MultiHeadAttention


class MockGroup:
    def __init__(self, world_size) -> None:
        self.world_size = world_size
        self.current_rank = 0

    def size(self):
        return self.world_size

    def rank(self):
        self.current_rank += 1
        return self.current_rank - 1


def _init_dist():
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "gloo", store=torch.distributed.HashStore(), rank=0, world_size=1
        )


def test_cp_attention_creation():
    """CPMultiHeadAttention can be created via to_cp() and has full (unsharded) weights."""
    _init_dist()
    emb_dim, nheads, kvheads = 256, 8, 4
    attention = MultiHeadAttention(
        emb_dim=emb_dim,
        emb_kq=emb_dim // nheads,
        emb_v=emb_dim // nheads,
        nheads=nheads,
        kvheads=kvheads,
        fused=True,
    )

    group = MockGroup(world_size=4)
    cp_attention = attention.to_cp(group)
    assert isinstance(cp_attention, CPMultiHeadAttention)
    # No head sharding: nheads and kvheads should be unchanged
    assert cp_attention.nheads == nheads
    assert cp_attention.kvheads == kvheads
    assert cp_attention.cp_rank == 0
    assert cp_attention.cp_world_size == 4


def test_cp_attention_same_architecture():
    """CP attention should have the same keys and weight shapes (no sharding)."""
    _init_dist()
    emb_dim, nheads, kvheads = 256, 8, 4
    attention = MultiHeadAttention(
        emb_dim=emb_dim,
        emb_kq=emb_dim // nheads,
        emb_v=emb_dim // nheads,
        nheads=nheads,
        kvheads=kvheads,
        fused=True,
    )
    orig_sd = attention.state_dict()

    group = MockGroup(world_size=4)
    cp_attention = attention.to_cp(group)
    cp_sd = cp_attention.state_dict()

    assert set(orig_sd.keys()) == set(cp_sd.keys())
    for key in orig_sd:
        assert orig_sd[key].shape == cp_sd[key].shape, (
            f"{key}: expected {orig_sd[key].shape}, got {cp_sd[key].shape}"
        )
