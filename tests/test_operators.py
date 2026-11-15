"""Small synthetic exactness regression; never substitutes for real training acceptance."""
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();w=a.work.resolve()
assert 'No running processes found' in subprocess.check_output(['ixsmi'],text=True),'GPU busy'
sys.path[:0]=[str(w/'ops'),str(w/'frontend/src'),str(w/'Megatron-LM')]
import torch
from bda_candidate import candidate as bda
from dispatch_candidate import candidate as swiglu,reference
from adam_candidate import run
from torch.utils.checkpoint import checkpoint
torch.manual_seed(1234);torch.cuda.set_device(0)
def equal(a,b):assert torch.equal(a,b),float((a-b).abs().max())
x=torch.randn((512,1,2560),device='cuda',dtype=torch.bfloat16,requires_grad=True);r=torch.randn_like(x,requires_grad=True);g=torch.randn_like(x)
y=bda(x,r,'r1');z=r+x;equal(y,z)
expected=torch.autograd.grad(z,(x,r),g)
for u,v in zip(torch.autograd.grad(y,(x,r),g),expected):equal(u,v)
recomputed=checkpoint(lambda a,b:bda(a,b,'r1'),x,r,use_reentrant=True);equal(recomputed,z);recomputed.backward(g)
for u,v in zip((x.grad,r.grad),expected):equal(u,v)
x=torch.randn((512,1,1600),device='cuda',dtype=torch.bfloat16,requires_grad=True);g=torch.randn((512,1,800),device='cuda',dtype=torch.bfloat16)
y=swiglu(x,'r3');z=reference(x);equal(y,z);expected=torch.autograd.grad(z,x,g)[0];equal(torch.autograd.grad(y,x,g)[0],expected)
recomputed=checkpoint(lambda t:swiglu(t,'r3'),x,use_reentrant=True);equal(recomputed,z);recomputed.backward(g);equal(x.grad,expected)
p0=torch.randn(8192,device='cuda',requires_grad=True);p1=p0.detach().clone()
opt=torch.optim.AdamW([p0],lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=True)
state=dict(step=torch.tensor(0.),exp_avg=torch.zeros_like(p1),exp_avg_sq=torch.zeros_like(p1))
for step in range(1,6):
    grad=torch.randn_like(p0);p0.grad=grad
    opt.step();run([dict(options=opt.param_groups[0],tensors=[dict(p=p1,g=grad,state=state)])],2)
    equal(p0,p1)
    for k in ['step','exp_avg','exp_avg_sq']:equal(opt.state[p0][k],state[k])
torch.cuda.synchronize();print(json.dumps(dict(passed=True,scope='synthetic BDA/SwiGLU F+B/recompute and5 AdamW updates; not TP8 training or GQA distributed revalidation')))
