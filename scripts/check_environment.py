import argparse,importlib,importlib.metadata,json,os,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);a=p.parse_args();w=a.work.resolve()
sys.path[:0]=[str(w/'ops'),str(w/'frontend/src'),str(w/'Megatron-LM')]
import torch,triton
assert torch.__version__.split('+')[0]=='2.10.0',torch.__version__
assert importlib.metadata.version('torch')=='2.10.0+corex.5.0.0','Requires vendor distribution metadata'
for n in ['adam_candidate','optimizer_adapter','bda_candidate','bda_adapter','mlp_adapter','dispatch_candidate','jiuge_training.gqa_adapter','jiuge_training.compute_adapter','jiuge_training.rope_candidate']:
    importlib.import_module(n)
from bda_adapter import SOURCE_COMPATIBLE as bda
from mlp_adapter import SOURCE_COMPATIBLE as mlp
from jiuge_training.gqa_adapter import SOURCE_COMPATIBLE as gqa
assert bda and mlp and gqa,'Pinned source guards must match'
print(json.dumps(dict(imports_pass=True,source_guards_pass=True,torch=torch.__version__,triton=triton.__version__,gpu_count=torch.cuda.device_count(),note='No training; no NVIDIA wheel installation'),indent=2))
