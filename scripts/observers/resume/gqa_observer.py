"""Diagnostic-only full state hashes; never reset checkpoint-restored RNG."""
import functools,hashlib,json,os,random
from pathlib import Path
import numpy as np
import torch
OUT=Path(os.environ['KDA_GQA_OUT']);RANK=int(os.environ['RANK']);MODE='numeric';STEP=0;MODELS=[];FIRST=True
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
def attach(model):MODELS.append(model)
def restore(s):
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    torch.set_rng_state(s['cpu']);torch.cuda.set_rng_state(s['cuda']);get_cuda_rng_tracker().set_states(s['tp']);random.setstate(s['python']);np.random.set_state(s['numpy'])
def rng():
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker
    return dict(cpu=digest(torch.get_rng_state()),cuda=digest(torch.cuda.get_rng_state()),tp={k:digest(v) for k,v in get_cuda_rng_tracker().get_states().items()},python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),numpy=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest())
def tensor(t):return dict(shape=list(t.shape),dtype=str(t.dtype),sha256=digest(t))
def snapshot(opt,label,scheduler,full=True):
    from megatron.training import get_args
    args=get_args();model={};grads={}
    for m in MODELS:
        for name,p in m.named_parameters():
            model[name]=tensor(p if full or 'layernorm' in name or name.startswith(('decoder.layers.0.','decoder.layers.61.')) else p.flatten()[:4096])
            if 'layernorm' in name or name.startswith(('decoder.layers.0.','decoder.layers.61.')):
                g=getattr(p,'main_grad',p.grad)
                if g is not None:grads[name]=tensor(g)
    children=[]
    for child in opt.chained_optimizers:
        steps=[s['step'] for s in child.optimizer.state.values() if 'step' in s]
        assert len({s.data_ptr() for s in steps})==len(steps),'Aliased Adam step counters after restore'
        groups=[]
        for g in child.optimizer.param_groups:
            ts=[]
            for p in g['params']:
                ts.append(dict(parameter=tensor(p if full else p.flatten()[:4096]),state={k:(tensor(v if full else v.flatten()[:4096])|({'value':v.item()} if v.numel()==1 else {})) if torch.is_tensor(v) else v for k,v in child.optimizer.state[p].items()}))
            groups.append(dict(options={k:v for k,v in g.items() if k!='params'},tensors=ts))
        children.append(dict(wrapper=str(type(child)),backend=str(type(child.optimizer)),groups=groups))
    value=dict(label=label,step=STEP,iteration=args.iteration,consumed_train_samples=args.consumed_train_samples,model=model,gradients=grads,optimizers=children,rng=rng(),scheduler=scheduler.state_dict())
    (OUT/f'{label}-rank{RANK}.json').write_text(json.dumps(value,indent=2,default=str))
def wrap_step(fn):
    @functools.wraps(fn)
    def wrapped(*args,**kwargs):
        global STEP,FIRST
        STEP=kwargs['iteration']+1;opt=args[3];scheduler=args[4]
        if FIRST:
            FIRST=False
            if os.environ['KDA_ADAM_MODE']=='r2':
                from optimizer_adapter import install
                install(opt,OUT,RANK)
            if os.environ.get('KDA_RESUME_LOAD','')=='':
                ref=Path(os.environ['KDA_INITIAL_REFERENCE']);restore(torch.load(ref/f'rng-rank{RANK}.pt',weights_only=False))
                assert scheduler.wd_incr_style=='constant'
                scheduler.wd_incr_steps=240  # Original15-update constant-WD schedule; extend only diagnostic loop.
            snapshot(opt,f'pre{STEP}',scheduler)
        elif STEP in (3,16):snapshot(opt,f'pre{STEP}',scheduler)
        result=fn(*args,**kwargs)
        if os.environ.get('KDA_VALIDATION_LONG')=='1':snapshot(opt,f'post{STEP}',scheduler,full=STEP in (25,50))
        elif STEP in (2,4,15,17):snapshot(opt,f'post{STEP}',scheduler)
        return result
    return wrapped
