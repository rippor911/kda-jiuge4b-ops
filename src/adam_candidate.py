"""Experimental FP32 AdamW fusion, not integrated into training."""
import triton
import triton.language as tl
from torch._inductor.runtime.triton_helpers import libdevice

@triton.jit
def update(P,G,M,V,N:tl.constexpr,DECAY:tl.constexpr,B1:tl.constexpr,B2:tl.constexpr,SS,BC,EPS:tl.constexpr,ROUND:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);ok=i<N
    p=tl.load(P+i,ok,0);g=tl.load(G+i,ok,0);m=tl.load(M+i,ok,0);v=tl.load(V+i,ok,0)
    p=p*DECAY
    m=tl.fma(1-B1,g-m,m)
    v=tl.fma(1-B2,g*g,v*B2)
    den=tl.sqrt(v)
    den=tl.div_rn(den,BC)+EPS
    ratio=tl.div_rn(m,den)
    p=tl.fma(SS,ratio,p)
    tl.store(P+i,p,ok);tl.store(M+i,m,ok);tl.store(V+i,v,ok)

def run(groups,round_id):
    for group in groups:
        o=group['options'];b1,b2=o['betas']
        for t in group['tensors']:
            st=t['state'];st['step'].add_(1);s=st['step'].item()
            update[(triton.cdiv(t['p'].numel(),1024),)](t['p'],t['g'],st['exp_avg'],st['exp_avg_sq'],t['p'].numel(),1-o['lr']*o['weight_decay'],b1,b2,-o['lr']/(1-b1**s),(1-b2**s)**.5,o['eps'],round_id,1024,enable_fp_fusion=False)
