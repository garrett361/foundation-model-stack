from torch import nn
import torch
from torch.distributed.distributed_c10d import ProcessGroup

from fms.modules.embedding import TPEmbedding, CPEmbedding
from fms.modules.positions import Alibi
from fms.distributed import rank_and_world
from fms.utils.generation import pad_input_ids
import torch.nn.functional as F
import torch.distributed._functional_collectives as funcol

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
    if model_input.size(-1)  > 1: 
        model_input_chunk = model_input.tensor_split(world_size, dim=-1)[rank]
        #for AIU handling padding
        #padding_height = (0,32)
        #model_input_chunk = F.pad(model_input_chunk_unpad, padding_height, "constant", 0)
        #pad_input_ids(model_input_chunk, min_pad_length=64-model_input_chunk.size(-1))
    else:
      model_input_chunk = model_input
    return model_input_chunk
    
def apply_gather_tensor_cp(model_input:torch.LongTensor, group: ProcessGroup):
    return funcol.all_gather_tensor(
            model_input.contiguous(),
            gather_dim=1,
            group=group)
