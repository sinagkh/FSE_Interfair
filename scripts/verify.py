"""Check archive integrity, expected evidence coverage, and checkpoint exclusion."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
# Created locally by the README commands; never part of the distributed archive.
LOCAL={'.venv','reproduced','retrained','__pycache__'}
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
manifest=json.loads((ROOT/'MANIFEST.json').read_text());errors=[]
for row in manifest['files']:
    p=ROOT/row['path']
    if not p.is_file() or p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:errors.append(row['path'])
assert not errors,errors
for p in ROOT.rglob('*'):
    if LOCAL.intersection(p.relative_to(ROOT).parts):continue
    if p.is_file() and p.suffix.lower() in ['.pt','.pth','.ckpt','.safetensors']:
        raise AssertionError(('Unexpected checkpoint',str(p)))
print('PASS:',len(manifest['files']),'manifest files verified; no checkpoints in the distributed artifact.')
