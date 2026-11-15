"""Correctness-only observer; disabled in formal performance runs."""
import functools, hashlib, json, os, random
from pathlib import Path
import numpy as np
import torch
OUT=Path(os.environ['KDA_GQA_OUT']); RANK=int(os.environ['RANK'])
BASE=Path(os.environ['KDA_GQA_REFERENCE']) if os.environ.get('KDA_GQA_REFERENCE') else None
MODE=os.environ['KDA_GQA_MODE']; STEP=0; MODELS=[]
def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
def canonical(obj):
    if torch.is_tensor(obj):
        return dict(shape=list(obj.shape),dtype=str(obj.dtype),sha256=hashlib.sha256(obj.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(obj,dict):return {str(k):canonical(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [canonical(v) for v in obj]
    return obj

def save(name,obj):
    if MODE=='numeric' and name!='rng':
        value=canonical(obj)
        (OUT/f'{name}-rank{RANK}.json').write_text(json.dumps(value,sort_keys=True))
        reference=canonical(torch.load(BASE/f'{name}-rank{RANK}.pt',map_location='cpu',weights_only=False,mmap=True))
        equal=value==reference
        with (OUT/f'numeric-compare-rank{RANK}.jsonl').open('a') as log:log.write(json.dumps(dict(record=name,byte_equal=equal))+'\n')
        assert equal, f'Numeric snapshot mismatch: {name} rank{RANK}'
    else:torch.save(obj,OUT/f'{name}-rank{RANK}.pt')
def attach(model):
    MODELS.append(model)
    if os.environ.get('KDA_SOFTMAX_MODE') in ('r1','r2'):
        from jiuge_training.softmax_adapter import BoundaryDotProductAttention
        types=[type(layer.self_attention.core_attention).__name__ for layer in model.decoder.layers]
        assert all(isinstance(layer.self_attention.core_attention,BoundaryDotProductAttention) for layer in model.decoder.layers),types
        (OUT/f'softmax-modules-rank{RANK}.json').write_text(json.dumps(types))
    if os.environ.get('KDA_MLP_MODE')=='r3':
        from mlp_adapter import DispatchMLP
        assert all(isinstance(layer.mlp,DispatchMLP) for layer in model.decoder.layers)
    hashes={n:digest(p) for n,p in model.named_parameters()}
    (OUT/f'initial-rank{RANK}.json').write_text(json.dumps(hashes,sort_keys=True))
    if BASE:assert hashes==json.loads((BASE/f'initial-rank{RANK}.json').read_text()),'Initial model mismatch'
    dispatch={n:('kv_pair' if m.gqa_pair is not None else 'full_tp') for n,m in model.named_modules() if hasattr(m,'gqa_pair')}
    (OUT/f'dispatch-rank{RANK}.json').write_text(json.dumps(dispatch))
    if os.environ['KDA_GQA_BACKEND']=='kv_pair' and os.environ.get('KDA_GQA_LAYOUT')=='tp8':
        assert len(dispatch)==62 and set(dispatch.values())=={'kv_pair'},dispatch
    if os.environ.get('KDA_SOFTMAX_MODE')=='capture' and RANK==0:
        for name,mod in model.named_modules():
            if not name.endswith('core_attention.scale_mask_softmax') or not name.startswith(('decoder.layers.0.','decoder.layers.31.','decoder.layers.61.')):continue
            def make_hook(name,mod):
                captured=False
                def hook(module,inputs,output):
                    nonlocal captured
                    if captured or not output.requires_grad:return
                    captured=True
                    x,mask=inputs[:2]
                    record=dict(name=name,x=x.detach().cpu(),mask=None if mask is None else mask.detach().cpu(),y=output.detach().cpu(),x_stride=list(x.stride()),y_stride=list(output.stride()),scale=mod.scale,input_in_bf16=mod.input_in_bf16,softmax_in_fp32=mod.softmax_in_fp32,attn_mask_type=str(mod.attn_mask_type),window_size=mod.window_size,softmax_offset=None if len(inputs)<3 or inputs[2] is None else inputs[2].detach().cpu())
                    target=OUT/('softmax-'+name+'.pt')
                    def grad_hook(g):
                        record['grad_output']=g.detach().cpu();record['grad_stride']=list(g.stride());torch.save(record,target)
                    output.register_hook(grad_hook)
                return hook
            mod.register_forward_hook(make_hook(name,mod))
    if RANK==0 and os.environ.get('KDA_MLP_DIAGNOSTIC')=='1':
        for name,mod in model.named_modules():
            if not name.endswith(('mlp.linear_fc1','mlp.linear_fc2')):continue
            def install(name,mod):
                scopes=[]
                captured=False
                def pre(module,args):
                    scopes.append(torch.profiler.record_function('TARGET::'+name))
                    scopes[-1].__enter__()
                    nonlocal captured
                    if captured or not args[0].requires_grad or not name.startswith(('decoder.layers.0.','decoder.layers.31.','decoder.layers.61.')):return
                    captured=True
                    x=args[0]
                    record=dict(name=name,x=x.detach().cpu(),stride=list(x.stride()),weight=module.weight.detach().cpu())
                    if name.endswith('linear_fc2'):
                        def grad(g):
                            record['grad_input']=g.detach().cpu();record['grad_stride']=list(g.stride())
                            torch.save(record,OUT/('capture-'+name+'.pt'))
                        x.register_hook(grad)
                    else:torch.save(record,OUT/('capture-'+name+'.pt'))
                def post(module,args,output):
                    if name.endswith('linear_fc1') and output[0].requires_grad and not hasattr(module,'_capture_done') and name.startswith(('decoder.layers.0.','decoder.layers.31.','decoder.layers.61.')):
                        module._capture_done=True
                        record=dict(name=name,y=output[0].detach().cpu(),stride=list(output[0].stride()))
                        torch.save(record,OUT/('swiglu-input-'+name+'.pt'))
                    scopes.pop().__exit__(None,None,None)
                mod.register_forward_pre_hook(pre);mod.register_forward_hook(post)
            install(name,mod)
    if MODE!='numeric':return
    seen=set()
    for name,m in model.named_modules():
        if not name.endswith(('self_attention.linear_qkv','self_attention.core_attention','mlp')):continue
        def hook(mod,inputs,output,name=name):
            if STEP!=1 or name in seen:return
            seen.add(name)
            y=output[0] if isinstance(output,tuple) else output
            record={'x':digest(inputs[0]),'y':digest(y),'xshape':list(inputs[0].shape),'yshape':list(y.shape)}
            with (OUT/f'forward-rank{RANK}.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,**record))+'\n')
        m.register_forward_hook(hook)
def selected():
    result={}
    for model in MODELS:
        for name,p in model.named_parameters():
            if 'layernorm' not in name and not (name.startswith(('decoder.layers.0.','decoder.layers.61.')) and any(k in name for k in ('linear_qkv','mlp.linear_fc1','mlp.linear_fc2'))):continue
            g=getattr(p,'main_grad',None)
            if g is None:g=p.grad
            result[name]={'parameter':p.detach().cpu().clone(),'gradient':None if g is None else g.detach().cpu().clone()}
    return result
def adam(opt):
    records=[]
    for gi,group in enumerate(opt.optimizer.param_groups):
        for pi,p in enumerate(group['params']):
            def sample(t):return t.detach().flatten()[:4096].cpu().clone() if torch.is_tensor(t) else t
            records.append(dict(group=gi,index=pi,shape=list(p.shape),parameter=sample(p),gradient=sample(p.grad),state={k:sample(v) for k,v in opt.optimizer.state.get(p,{}).items()}))
    return records
def rng():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(),tp=get_cuda_rng_tracker().get_states(),python=random.getstate(),numpy=np.random.get_state())
def restore(s):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda']);get_cuda_rng_tracker().set_states(s['tp']);random.setstate(s['python']);np.random.set_state(s['numpy'])
def wrap_step(fn):
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        global STEP
        STEP+=1
        if STEP==1 and os.environ.get('KDA_ADAM_MODE')=='r2':
            from optimizer_adapter import install
            install(args[3] if len(args)>3 else kwargs['optimizer'],OUT,RANK)
        if STEP==1:
            if BASE:restore(torch.load(BASE/f'rng-rank{RANK}.pt',weights_only=False))
            save('rng',rng())
        if os.environ.get('KDA_OPT_AUDIT'):
            from optimizer_audit import invoke
            return invoke(fn,args,kwargs,STEP,OUT,RANK,MODELS)
        if MODE!='numeric':
            if os.environ.get('KDA_LIGHT_PROFILE')=='1' and STEP==3 and RANK==0:
                import gzip,shutil,time
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_stack=False,profile_memory=False) as prof:
                    result=fn(*args,**kwargs)
                raw=OUT/'light-rank0.json';prof.export_chrome_trace(str(raw))
                with raw.open('rb') as src,gzip.open(str(raw)+'.gz','wb') as dst:shutil.copyfileobj(src,dst)
                raw.unlink()
                (OUT/'LIGHT_PROFILE_METADATA.json').write_text(json.dumps(dict(profiled=True,rank=0,update=3,record_shapes=True,with_stack=False,warning='Diagnostic only. StepRecorder profiler field describes native profiling, not this external context.')))
                return result
            return fn(*args,**kwargs)
        opt=args[3] if len(args)>3 else kwargs['optimizer']
        original=opt.step; inner=opt.optimizer.step
        def observed(*a,**kw):
            save(f'pre-step{STEP}',selected())
            result=original(*a,**kw)
            save(f'post-step{STEP}',selected())
            return result
        def observed_inner(*a,**kw):
            save(f'adam-before-step{STEP}',adam(opt));result=inner(*a,**kw);save(f'adam-after-step{STEP}',adam(opt));return result
        opt.step=observed;opt.optimizer.step=observed_inner
        try:return fn(*args,**kwargs)
        finally:opt.step=original;opt.optimizer.step=inner
    return wrapped
