"""GQA assembly backend. Reuses pinned gather autograd (backward reduce-scatter)."""
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from megatron.core.tensor_parallel.mappings import all_gather_last_dim_from_tensor_parallel_region
_PAIR = None

def pair_group(tp):
    global _PAIR
    if _PAIR is not None:
        return _PAIR
    ranks = dist.get_process_group_ranks(tp)
    assert len(ranks) == 8 and dist.get_world_size() == 8
    mappings = [None] * 8
    dist.all_gather_object(mappings, ranks)
    assert all(m == ranks for m in mappings), mappings
    mine = dist.get_rank()
    pairs = [ranks[i:i+2] for i in range(0, 8, 2)]
    for pair in pairs:
        group = dist.new_group(ranks=pair, backend=dist.get_backend(tp))
        if mine in pair:
            _PAIR = group
            assert dist.get_rank(group) == ranks.index(mine) % 2
    out = os.environ.get('KDA_GQA_OUT')
    if out:
        Path(out, f'group-rank{mine}.json').write_text(json.dumps(dict(global_rank=mine, pid=os.getpid(), cuda_device=torch.cuda.current_device(), device_name=torch.cuda.get_device_name(),
            tp_local_rank=dist.get_rank(tp), tp_global_ranks=ranks, pairs=pairs,
            pair_global_ranks=dist.get_process_group_ranks(_PAIR),
            local_rank=int(os.environ.get('LOCAL_RANK', -1)),
            cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))))
    return _PAIR

def assemble(x, group):
    assert x.shape[-1] == 384 and group.size() == 2
    return all_gather_last_dim_from_tensor_parallel_region(x, group=group)
