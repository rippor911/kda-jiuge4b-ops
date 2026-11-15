"""Guarded ModuleSpec residual-add dispatch; native fallback remains explicit."""
import atexit,hashlib,json,os
from pathlib import Path
import torch
import megatron.core.fusions.fused_bias_dropout as native

SOURCE_COMPATIBLE=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()=='897e60739d918ec917f7a8bfc578ac82a5a413f3bb80314a747a18d69c7b95df'
COUNTS={'optimized':0,'fallback':0};CAPTURED=set()
def report():
    if os.environ.get('KDA_GQA_OUT'):Path(os.environ['KDA_GQA_OUT'],f'bda-rank{os.environ["RANK"]}.json').write_text(json.dumps(COUNTS))
atexit.register(report)
def factory(kind,training,fused):
    fallback=native.get_bias_dropout_add(training,fused)
    def apply(pair,residual,prob):
        x,bias=pair;mode=os.environ.get('KDA_BDA_MODE','off')
        supported=(SOURCE_COMPATIBLE and training and fused and prob==0 and bias is None and os.environ.get('KDA_GQA_LAYOUT')=='tp8'
            and x.is_cuda and residual.is_cuda and x.device==residual.device and x.dtype==residual.dtype==torch.bfloat16
            and tuple(x.shape)==tuple(residual.shape)==(512,1,2560) and x.is_contiguous() and residual.is_contiguous())
        if supported and mode=='r1':
            from bda_candidate import candidate
            COUNTS['optimized']+=1
            return candidate(x,residual,mode)
        COUNTS['fallback']+=1;y=fallback(pair,residual,prob)
        if mode=='capture' and os.environ['RANK']=='0' and torch.is_grad_enabled() and kind not in CAPTURED:
            def save(g):
                if kind in CAPTURED:return
                CAPTURED.add(kind)
                payload={'x':x.detach().cpu().clone(),'residual':residual.detach().cpu().clone(),'output':y.detach().cpu().clone(),
                    'x_stride':list(x.stride()),'residual_stride':list(residual.stride()),'prob':prob,'training':training,'bias':None if bias is None else bias.detach().cpu().clone(),
                    'native_source_sha256':hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest()}
                payload['grad_output']=g.detach().cpu().clone();torch.save(payload,Path(os.environ['KDA_GQA_OUT'])/f'bda-{kind}.pt')
            y.register_hook(save)
        return y
    return apply
def attention_bda(training,fused):return factory('attention',training,fused)
def mlp_bda(training,fused):return factory('mlp',training,fused)
