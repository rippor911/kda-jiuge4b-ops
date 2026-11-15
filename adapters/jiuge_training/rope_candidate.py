"""Codex round1: native trig plus fused FP32 RoPE forward/backward, no FMA."""
import torch
import triton
import triton.language as tl
import weakref

_CACHE = {}
CACHE_STATS = {'hits': 0, 'misses': 0}

@triton.jit
def _forward(X,C,S,Y,N:tl.constexpr,H:tl.constexpr,S0:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    d=i%64;h=(i//64)%H;seq=i//(H*64);other=(d+32)%64
    x=tl.load(X+seq*S0+h*S2+d*S3,valid,0)
    z=tl.load(X+seq*S0+h*S2+other*S3,valid,0)
    c=tl.load(C+seq*64+d,valid,0);s=tl.load(S+seq*64+d,valid,0)
    z=tl.where(d<32,-z,z)
    a=tl.inline_asm_elementwise("ml_mul_f32 $0, $1, $2",constraints="=r,r,r",args=[x,c],dtype=tl.float32,is_pure=True,pack=1)
    b=tl.inline_asm_elementwise("ml_mul_f32 $0, $1, $2",constraints="=r,r,r",args=[z,s],dtype=tl.float32,is_pure=True,pack=1)
    tl.store(Y+i,a+b,valid)

@triton.jit
def _backward(G,C,S,DX,N:tl.constexpr,H:tl.constexpr,S0:tl.constexpr,S2:tl.constexpr,S3:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    d=i%64;h=(i//64)%H;seq=i//(H*64);other=(d+32)%64
    g=tl.load(G+seq*S0+h*S2+d*S3,valid,0)
    gj=tl.load(G+seq*S0+h*S2+other*S3,valid,0)
    c=tl.load(C+seq*64+d,valid,0);sj=tl.load(S+seq*64+other,valid,0)
    a=g*c;b=gj*sj;b=tl.where(d<32,b,-b)
    tl.store(DX+i,a+b,valid)

class _RoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,c,s):
        y=torch.empty(x.shape,device=x.device,dtype=x.dtype)
        _forward[(triton.cdiv(x.numel(),256),)](x,c,s,y,x.numel(),x.shape[2],x.stride(0),x.stride(2),x.stride(3),256,num_warps=4,enable_fp_fusion=False)
        ctx.save_for_backward(c,s);return y
    @staticmethod
    def backward(ctx,g):
        c,s=ctx.saved_tensors;dx=torch.empty(g.shape,device=g.device,dtype=g.dtype)
        _backward[(triton.cdiv(g.numel(),256),)](g,c,s,dx,g.numel(),g.shape[2],g.stride(0),g.stride(2),g.stride(3),256,num_warps=4,enable_fp_fusion=False)
        return dx,None,None

def rope(x,freqs,round='r1'):
    assert round in ('r1','r2')
    key=(id(freqs),freqs._version,x.dtype,torch.cuda.current_stream(x.device).cuda_stream)
    entry=_CACHE.get(key) if round=='r2' else None
    if entry is not None and entry[0]() is freqs:
        c,s=entry[1:];CACHE_STATS['hits']+=1
    else:
        c=(torch.cos(freqs)*1.).to(x.dtype).contiguous()
        s=(torch.sin(freqs)*1.).to(x.dtype).contiguous()
        if round=='r2':
            CACHE_STATS['misses']+=1
            if len(_CACHE)>=16:_CACHE.clear()
            _CACHE[key]=(weakref.ref(freqs),c,s)
    return _RoPE.apply(x,c,s)
