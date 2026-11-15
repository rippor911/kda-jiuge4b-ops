"""R1 host dispatch: reuse native Torch addition and its autograd implementation."""
def candidate(x,residual,mode):
    assert mode=='r1'
    return residual+x
