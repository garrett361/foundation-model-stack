from torch import nn
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from fms.modules.embedding import TPEmbedding, CPEmbedding
from fms.modules.positions import Alibi
from fms.distributed import rank_and_world


# this probably belongs somewhere else but can't go in fms.distribtued b/c
# circular dependency.
def _cp_wrapped(module: nn.Module, group: ProcessGroup):
    if hasattr(module, "to_cp") and callable(module.to_cp):
        return module.to_cp(group)
    return module

def apply_layer_cp(model: nn.Module, group: ProcessGroup):
    wrapped = _cp_wrapped(model, group)
    if wrapped is not model:
        return wrapped

    for name, layer in model.named_children():
        cp_layer = apply_layer_cp(layer, group)
        setattr(model, name, cp_layer)
    return model

def apply_input_cp(model_input:torch.LongTensor, group: ProcessGroup):
    rank,world_size = rank_and_world(group)
    #print(model_input.size())
    if model_input.size(-1)  > 1: 
        model_input_chunk = model_input.tensor_split(world_size, dim=-1)[rank]
    else:
        model_input_chunk = model_input
    return model_input_chunk
    

