"""Explicit ModuleSpec MLP adapter; only compiled-native SwiGLU dispatch changes."""
import atexit,hashlib,json,os
from pathlib import Path
import torch
import torch.nn.functional as F
import megatron.core.transformer.mlp as native_module
from megatron.core.transformer.mlp import MLP
from megatron.core.typed_torch import apply_module
from megatron.core.utils import nvtx_range_push,nvtx_range_pop
from dispatch_candidate import candidate

SOURCE_COMPATIBLE=hashlib.sha256(Path(native_module.__file__).read_bytes()).hexdigest()=='f54a7209db5cad4e095143cbbc1eedcbf41e913ea1eec64db3cb1b8408e6ff07'
COUNTS={'optimized':0,'fallback':0}
def report():
    if os.environ.get('KDA_GQA_OUT'):
        Path(os.environ['KDA_GQA_OUT'],f'mlp-rank{os.environ["RANK"]}.json').write_text(json.dumps(COUNTS))
atexit.register(report)

class DispatchMLP(MLP):
    def forward(self,hidden_states,per_token_scale=None,**kwargs):
        c=self.config
        supported=(SOURCE_COMPATIBLE and os.environ.get('KDA_MLP_MODE')=='r3' and c.tensor_model_parallel_size==8
            and not (set(kwargs)-{'padding_mask'}) and per_token_scale is None and not c.use_te_activation_func and c.bias_activation_fusion
            and self.activation_func is F.silu and c.gated_linear_unit and not c.add_bias_linear
            and not c.activation_func_fp8_input_store and not c.cpu_offloading
            and hidden_states.is_cuda and hidden_states.dtype==torch.bfloat16 and tuple(hidden_states.shape)==(512,1,2560)
            and hidden_states.is_contiguous() and tuple(self.linear_fc1.weight.shape)==(1600,2560)
            and tuple(self.linear_fc2.weight.shape)==(2560,800))
        if not supported:
            COUNTS['fallback']+=1
            return super().forward(hidden_states,per_token_scale,**kwargs)
        nvtx_range_push(suffix='linear_fc1')
        x,bias=apply_module(self.linear_fc1)(hidden_states)
        nvtx_range_pop(suffix='linear_fc1')
        assert bias is None
        nvtx_range_push(suffix='activation')
        y=candidate(x,'r3')
        nvtx_range_pop(suffix='activation')
        COUNTS['optimized']+=1
        nvtx_range_push(suffix='linear_fc2')
        result=apply_module(self.linear_fc2)(y)
        nvtx_range_pop(suffix='linear_fc2')
        return result
