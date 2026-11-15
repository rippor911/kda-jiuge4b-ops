"""Explicit RoPE backend dispatch and representative-input capture."""
import os, atexit, json
from pathlib import Path
import torch
from megatron.core.models.common.embeddings.rope_utils import apply_rotary_pos_emb
MODE=os.environ.get('KDA_COMPUTE_MODE','off')
SEEN=set(); GRAD_SEEN=set()
COUNTS={'optimized':0,'fallback':0}
def _report():
    if MODE not in ('r1','r2'):return
    from .rope_candidate import CACHE_STATS
    Path(os.environ['KDA_GQA_OUT'],f'compute-rank{os.environ["RANK"]}.json').write_text(json.dumps(dict(mode=MODE,dispatch=COUNTS,cache=CACHE_STATS)))
atexit.register(_report)
def dispatch(t,freqs,layer=None,**kwargs):
    if MODE=='capture':
        result=apply_rotary_pos_emb(t,freqs,**kwargs)
        rank=int(os.environ['RANK']);key=(layer,t.shape[-2])
        if rank==0 and layer in (1,32,62):
            out=Path(os.environ['KDA_GQA_OUT']);name=f'rope-layer{layer}-heads{t.shape[-2]}'
            if key not in SEEN:
                SEEN.add(key)
                torch.save(dict(x=t.detach().cpu(),freqs=freqs.detach().cpu(),y=result.detach().cpu(),
                    x_stride=list(t.stride()),freq_stride=list(freqs.stride()),y_stride=list(result.stride()),
                    mscale=kwargs.get('mscale',1.),rotary_interleaved=kwargs['config'].rotary_interleaved,
                    dtype=str(t.dtype)),out/(name+'.pt'))
            if result.requires_grad and key not in GRAD_SEEN:
                GRAD_SEEN.add(key)
                result.register_hook(lambda g,path=out/(name+'-g.pt'):torch.save(g.detach().cpu(),path))
        return result
    if MODE in ('r1','r2'):
        from .rope_candidate import rope
        cfg=kwargs['config']
        supported=(cfg.tensor_model_parallel_size==8 and os.environ.get('KDA_GQA_BACKEND')=='kv_pair'
            and t.is_cuda and t.dtype==torch.float32 and t.ndim==4 and t.shape[1]==1
            and t.shape[-1]==64 and freqs.shape==(t.shape[0],1,1,64) and freqs.dtype==torch.float32
            and not freqs.requires_grad and not cfg.rotary_interleaved and not cfg.apply_rope_fusion
            and not kwargs.get('mla_rotary_interleaved',False) and kwargs.get('cu_seqlens') is None
            and kwargs.get('mscale',1.)==1. and kwargs['cp_group'].size()==1)
        if supported:
            COUNTS['optimized']+=1
            return rope(t,freqs,round=MODE)
        COUNTS['fallback']+=1
    return apply_rotary_pos_emb(t,freqs,**kwargs)
