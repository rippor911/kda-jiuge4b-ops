"""Materialize an external runtime, without altering source checkouts or this repository."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def prepare(jiuge,megatron,work):
    lock=json.loads((ROOT/'configs/dependencies.lock.json').read_text())
    assert not work.is_relative_to(ROOT),'Runtime must be outside the Git repository'
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=megatron,text=True).strip()==lock['megatron_commit']
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=megatron,text=True),'Use clean pinned Megatron'
    for name,digest in lock['baseline_python_files'].items():
        assert hashlib.sha256((jiuge/name).read_bytes()).hexdigest()==digest,('Wrong archived frontend',name)
    work.mkdir(parents=True,exist_ok=False)
    frontend=work/'frontend';frontend.mkdir()
    shutil.copytree(jiuge/'src',frontend/'src',ignore=shutil.ignore_patterns('__pycache__'))
    # Archived originals use CRLF; the validated patched pair uses LF.
    for name in ['model.py','train.py']:
        f=frontend/'src/jiuge_training'/name;f.write_text(f.read_text())
    subprocess.run(['git','apply','--unidiff-zero',str(ROOT/'patches/jiuge-b6.patch')],cwd=frontend,check=True)
    shutil.copytree(ROOT/'adapters/jiuge_training',frontend/'src/jiuge_training',dirs_exist_ok=True)
    for name,digest in lock['prepared_python_files'].items():
        assert hashlib.sha256((frontend/name).read_bytes()).hexdigest()==digest,('Runtime differs from validated release',name)
    subprocess.run(['git','clone','--shared',str(megatron),str(work/'Megatron-LM')],check=True)
    subprocess.run(['git','apply',str(ROOT/'patches/megatron-checkpoint-compat.patch')],cwd=work/'Megatron-LM',check=True)
    shutil.copytree(ROOT/'src',work/'ops')
    (work/'PREPARED.json').write_text(json.dumps(dict(passed=True,frontend_files=len(lock['prepared_python_files']),megatron_commit=lock['megatron_commit']),indent=2))
    print('Prepared runtime matches validated frontend; checkpoint patch applied')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--jiuge',type=Path,required=True);p.add_argument('--megatron',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    prepare(a.jiuge.resolve(),a.megatron.resolve(),a.work.resolve())
