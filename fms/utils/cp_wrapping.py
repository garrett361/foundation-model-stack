from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup

from fms.modules.embedding import TPEmbedding, CPEmbedding
from fms.modules.positions import Alibi


# this probably belongs somewhere else but can't go in fms.distribtued b/c
# circular dependency.
def _cp_wrapped(module: nn.Module, group: ProcessGroup):
    if hasattr(module, "to_cp") and callable(module.to_cp):
        return module.to_cp(group)
    elif isinstance(module, Alibi):
        raise NotImplementedError("TODO: implement CP for Alibi")
        # tp_layer = TPAlibi.import_module(layer, world_size, rank, dtype)
        # setattr(model, name, tp_layer)
    elif isinstance(module, nn.Embedding):
        # We can't directly modify torch.nn modules to add the to_tp function
        return CPEmbedding.import_module(module, group)
    else:
        return module


def apply_cp(model: nn.Module, group: ProcessGroup):
    wrapped = _cp_wrapped(model, group)
    if wrapped is not model:
        return wrapped

    for name, layer in model.named_children():
        cp_layer = apply_cp(layer, group)
        setattr(model, name, cp_layer)
    return model
