"""Scan allowlist or all reachable Git blobs. Prints only locations, never secrets."""
import argparse,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--history',action='store_true');a=p.parse_args()
patterns={
    'private-key':r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
    'credential':r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[A-Z0-9]{16})',
    'credential-assignment':r'(?i)(?:password|api_key|access_token|secret_key)\s*[:=]\s*[\x22\x27][^\x22\x27\s]{6,}',
    'private-server-path':r'/(?:private|home|root)/[A-Za-z0-9_-]+',
    'private-ip':r'\b(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)\b',
    'private-phone':r'(?<!\d)1[3-9]\d{9}(?!\d)',
}
names=(ROOT/'MANIFEST.txt').read_text().splitlines();items=[]
if a.history:
    for line in subprocess.check_output(['git','rev-list','--objects','--all'],cwd=ROOT,text=True).splitlines():
        obj,_,name=line.partition(' ')
        if subprocess.check_output(['git','cat-file','-t',obj],cwd=ROOT,text=True).strip()=='blob':items.append((name,subprocess.check_output(['git','cat-file','blob',obj],cwd=ROOT)))
else:items=[(n,(ROOT/n).read_bytes()) for n in names]
issues=[]
for name,data in items:
    if len(data)>1024*1024 or b'\x00' in data:issues.append(dict(file=name,rule='binary-or-large'));continue
    if name.endswith(('.pt','.bin','.safetensors','.jsonl','.log','.so','.zip','.gz')):issues.append(dict(file=name,rule='excluded-type'))
    text=data.decode('utf-8')
    for key,pattern in patterns.items():
        for m in re.finditer(pattern,text):issues.append(dict(file=name,line=text.count('\n',0,m.start())+1,rule=key))
print(json.dumps(dict(passed=not issues,files=len(items),issues=issues,limitations='Heuristic scan plus manual review; does not grant redistribution permission'),indent=2));raise SystemExit(bool(issues))
