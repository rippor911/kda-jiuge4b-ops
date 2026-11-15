"""Compare runtime trajectories and saved diagnostic hashes without changing tolerance."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path);a=p.parse_args()
maximum=0.;snapshots=0
def state(s,grad=True):
    groups=[c['groups'] for c in s['optimizers']]
    for c in groups:
        for g in c:g['options'].pop('step',None)
    keys=['model','rng','scheduler']+(['gradients'] if grad else [])
    return {k:s[k] for k in keys}|dict(optimizers=groups)
for rank in range(8):
    x={s['step']:s for s in json.loads((a.baseline/f'steps-rank{rank}.json').read_text())['steps']}
    ys=json.loads((a.candidate/f'steps-rank{rank}.json').read_text())['steps'];assert ys
    for y in ys:
        z=x[y['step']];assert z['loss'] and z['loss']==y['loss'] and z['global_grad_norm']==y['global_grad_norm']
        assert z['completed'] and y['completed'] and not z['skipped_update'] and not y['skipped_update']
    first=a.candidate/f"pre{ys[0]['step']}-rank{rank}.json"
    if first.exists():assert state(json.loads(first.read_text()),False)==state(json.loads((a.baseline/first.name).read_text()),False)
    d=a.candidate/f'data-rank{rank}.jsonl'
    if d.exists():
        baseline=[json.loads(x) for x in (a.baseline/d.name).read_text().splitlines()]
        candidate=[json.loads(x) for x in d.read_text().splitlines()]
        assert [x for x in baseline if x['step'] in {y['step'] for y in ys}]==candidate
    for f in a.candidate.glob(f'post*-rank{rank}.json'):
        other=a.baseline/f.name;assert other.exists(),f.name
        assert state(json.loads(f.read_text()))==state(json.loads(other.read_text())),f.name;snapshots+=1
print(json.dumps(dict(passed=True,max_loss_delta=maximum,snapshot_pairs=snapshots,scope='Exact stored fields; see docs for snapshot coverage')))
