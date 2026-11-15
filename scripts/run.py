"""Portable path/config orchestration. Frozen math and guards remain unchanged."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',choices=['B0','B5','B6'],required=True);p.add_argument('--mode',choices=['formal','diagnostic'],default='formal');p.add_argument('--steps',type=int,default=15)
    p.add_argument('--load',type=Path);p.add_argument('--save',action='store_true');p.add_argument('--rng-reference',type=Path);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    work=a.work.resolve();out=a.out.resolve()
    assert not out.is_relative_to(ROOT),'Outputs must remain outside repository'
    assert (work/'PREPARED.json').exists()
    if a.mode=='formal':assert a.steps==15 and not a.load and not a.save,'Formal10+5 only; resume uses diagnostic mode'
    if a.mode=='diagnostic' and not a.load:assert a.rng_reference,'Fresh diagnostic needs RNG files from matching formal initialization'
    assert 1<=a.steps<=50
    cfg=json.loads((ROOT/'configs/tp8-cpt.json').read_text());cfg['model']['path']=str(a.model.resolve());t=cfg['task']
    t.update(data_path=str(a.data.resolve()),save=str(out/'checkpoints'),save_checkpoint=a.save,save_interval=a.steps,train_iters=50 if a.mode=='diagnostic' else 15,load=str(a.load.resolve()) if a.load else '')
    sys.path.insert(0,str(work/'frontend/src'))
    from jiuge_training.config import command,runtime_environment,prepare_checkpoint_directory
    env=runtime_environment(work/'Megatron-LM');env={k:v for k,v in env.items() if not k.startswith('KDA_')}
    obs='resume' if a.mode=='diagnostic' else 'formal'
    env['PYTHONPATH']=os.pathsep.join(map(str,[ROOT/'scripts/observers'/obs,work/'ops',work/'frontend/src',work/'Megatron-LM']))
    enabled=a.backend!='B0'
    env.update(KDA_GQA_OUT=str(out),KDA_GQA_BACKEND='kv_pair' if enabled else 'full_tp',KDA_GQA_MODE='formal',KDA_GQA_LAYOUT='tp8',KDA_COMPUTE_MODE='r2' if enabled else 'off',KDA_MLP_MODE='r3' if enabled else 'off',KDA_ADAM_MODE='r2' if enabled else 'off',KDA_BDA_MODE='r1' if a.backend=='B6' else 'off',KDA_SOFTMAX_MODE='off',KDA_LIGHT_PROFILE='0',KDA_RESUME_LOAD=str(a.load.resolve()) if a.load else '',KDA_INITIAL_REFERENCE=str(a.rng_reference.resolve()) if a.rng_reference else '',KDA_VALIDATION_LONG='1' if a.mode=='diagnostic' and a.steps==50 else '0',CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',CUDA_DEVICE_MAX_CONNECTIONS='1',OMP_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',USE_TF='0')
    path=out/'config.json';argv=[sys.executable,'-m','torch.distributed.run']+command(cfg,path)[1:]+['--no-gradient-accumulation-fusion']
    if a.mode=='diagnostic':argv+=['--lr-decay-iters','15','--exit-interval',str(a.steps)]
    plan=dict(config=cfg,argv=argv,backend=a.backend,mode=a.mode)
    if a.dry_run:print(json.dumps(plan,indent=2));return
    if a.save:
        parent=out.parent
        while not parent.exists():parent=parent.parent
        assert shutil.disk_usage(parent).free>57848410075+2*2**30,'Need a full checkpoint plus storage margin'
    status=subprocess.check_output(['ixsmi'],text=True);assert 'No running processes found' in status,'GPU busy; do not launch'
    assert a.model.is_dir() and a.data.is_dir()
    out.mkdir(parents=True,exist_ok=False);path.write_text(json.dumps(cfg,indent=2));(out/'launch.json').write_text(json.dumps(plan,indent=2));(out/'gpu-before.txt').write_text(status)
    if a.save:prepare_checkpoint_directory(cfg)
    def sha(f):
        h=hashlib.sha256()
        with f.open('rb') as s:
            for b in iter(lambda:s.read(8388608),b''):h.update(b)
        return h.hexdigest()
    (out/'data-fingerprints.json').write_text(json.dumps({f.name:sha(f) for f in sorted(a.data.iterdir()) if f.is_file()},indent=2))
    with (out/'training.log').open('w') as log:r=subprocess.run(argv,env=env,cwd=work/'frontend',stdout=log,stderr=subprocess.STDOUT)
    (out/'execution.json').write_text(json.dumps(dict(returncode=r.returncode)));assert r.returncode==0
    start=int((a.load/'latest_checkpointed_iteration.txt').read_text()) if a.load else 0
    for rank in range(8):
        s=json.loads((out/f'steps-rank{rank}.json').read_text());assert len(s['steps'])==a.steps-start
        assert all(x['completed'] and not x['skipped_update'] for x in s['steps'])
        assert s['optimizer_state']['passed'] and s['parameter_updates']['passed']
        if enabled:
            c=json.loads((out/f'optimizer-backend-0-rank{rank}.json').read_text());assert c['optimized']==a.steps-start-(0 if a.load else 1)
    print('All8 updates passed')
if __name__=='__main__':main()
