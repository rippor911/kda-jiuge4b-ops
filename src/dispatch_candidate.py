"""Experimental guarded host adapter around unchanged native compiler output."""
import torch
import hashlib,importlib.util,json
from pathlib import Path
from megatron.core.fusions.fused_bias_swiglu import bias_swiglu_impl
_MODULES={}
def modules(device):
    if device not in _MODULES:
        root=Path(__file__).resolve().parent
        contract=json.loads((root/'DISPATCH_CONTRACT.json').read_text())
        result=[]
        for kind in ('forward','backward'):
            source=root/f'native_{kind}.py'
            assert hashlib.sha256(source.read_bytes()).hexdigest()==contract['sources'][kind]['sha256']
            # Bind the unchanged generated arithmetic to this process's device.
            text=source.read_text().replace('index=0,','index='+str(device)+',').replace('_DeviceGuard(0)','_DeviceGuard('+str(device)+')').replace('set_device(0)','set_device('+str(device)+')').replace('get_raw_stream(0)','get_raw_stream('+str(device)+')')
            target=root/'device-kernels'/str(device)/source.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text)
            spec=importlib.util.spec_from_file_location('kda_native_'+kind+'_'+str(device),target)
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);result.append(module)
        _MODULES[device]=tuple(result)
    return _MODULES[device]

class Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,mode):
        ctx.save_for_backward(x);ctx.mode=mode
        fwd,bwd=modules(x.device.index)
        if mode=='r1':return fwd.call([x])[0]
        out=torch.empty((512,800),device=x.device,dtype=x.dtype)
        fwd.triton_poi_fused_mul_silu_split_0.run(x,out,409600,stream=torch._C._cuda_getCurrentRawStream(x.device.index))
        return out
    @staticmethod
    def backward(ctx,g):
        x,=ctx.saved_tensors
        fwd,bwd=modules(x.device.index)
        if ctx.mode=='r1':return bwd.call([x,g.contiguous()])[0],None
        dx=torch.empty_like(x)
        bwd.triton_poi_fused_add_cat_mul_rsub_sigmoid_silu_split_0.run(g.contiguous(),x,dx,819200,stream=torch._C._cuda_getCurrentRawStream(x.device.index))
        return dx,None

def reference(x):return bias_swiglu_impl(x,None,False,False)

def candidate(x,mode='r1'):
    if x.device.type!='cuda' or x.dtype!=torch.bfloat16 or tuple(x.shape)!=(512,1,1600) or not x.is_contiguous():return reference(x)
    return Dispatch.apply(x.view(512,1600),mode).view(512,1,800)
