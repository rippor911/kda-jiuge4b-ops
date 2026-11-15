"""Explicit replacement of the private optimizer backend; native state schema."""
import atexit,json,os
from pathlib import Path
import torch
from adam_candidate import run

class ExactAdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self,closure=None):
        valid=closure is None and os.environ.get('KDA_GQA_LAYOUT')=='tp8'
        groups=[]
        for g in self.param_groups:
            valid &= (g['betas']==(0.9,0.999) and g['eps']==1e-8 and g['lr']==1e-5 and g['weight_decay'] in (0.,.01) and g.get('decoupled_weight_decay',True) and not any(g.get(k,False) for k in ['amsgrad','maximize','capturable','differentiable','fused']))
            tensors=[]
            for p in g['params']:
                st=self.state[p]
                valid &= (p.dtype==torch.float32 and p.is_cuda and p.is_contiguous() and p.grad is not None and p.grad.dtype==torch.float32 and p.grad.is_contiguous() and set(st)=={'step','exp_avg','exp_avg_sq'})
                if set(st)=={'step','exp_avg','exp_avg_sq'}:
                    valid &= all(st[k].dtype==torch.float32 and st[k].device==p.device and st[k].is_contiguous() and st[k].shape==p.shape for k in ['exp_avg','exp_avg_sq'])
                    valid &= st['step'].device.type=='cpu' and st['step'].dtype==torch.float32 and st['step'].numel()==1
                tensors.append(dict(p=p,g=p.grad,state=st))
            groups.append(dict(options=g,tensors=tensors))
        if not valid:
            self.audit['fallback']+=1
            return super().step(closure)
        self.audit['optimized']+=1
        run(groups,2)
        return None

def install(outer,out,rank):
    for index,child in enumerate(outer.chained_optimizers):
        native=child.optimizer
        assert type(native) is torch.optim.AdamW
        assert not native._optimizer_step_pre_hooks and not native._optimizer_step_post_hooks
        replacement=ExactAdamW(native.param_groups)
        replacement.state=native.state;replacement.defaults=native.defaults
        replacement.audit=dict(optimized=0,fallback=0,type=str(type(native)),state_schema_unchanged=True)
        child.optimizer=replacement
        def save(replacement=replacement,index=index):
            (Path(out)/f'optimizer-backend-{index}-rank{rank}.json').write_text(json.dumps(replacement.audit,indent=2))
        atexit.register(save)
