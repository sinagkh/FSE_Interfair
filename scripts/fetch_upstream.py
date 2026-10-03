"""Fetch pinned optional baseline source files and verify every downloaded byte."""
from pathlib import Path
import argparse,json,hashlib,urllib.request,io,zipfile
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',choices=['cot','ltdd','mirrorfair'],required=True);a=p.parse_args()
entries=[e for e in json.loads((ROOT/'third_party/fetch_manifest.json').read_text())['entries'] if e['source']==a.source]
cache={}
for e in entries:
    if e['url'] not in cache:
        with urllib.request.urlopen(e['url'],timeout=60) as response:cache[e['url']]=response.read()
    blob=cache[e['url']]
    if 'archive_sha256' in e:
        assert hashlib.sha256(blob).hexdigest()==e['archive_sha256']
        with zipfile.ZipFile(io.BytesIO(blob)) as z:blob=z.read(e['member'])
    assert hashlib.sha256(blob).hexdigest()==e['sha256']
    dest=ROOT/e['destination'];assert (ROOT/'interfair').resolve() in dest.resolve().parents
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_bytes()==blob,'Existing local file differs; refusing to overwrite it.'
    else:dest.write_bytes(blob)
    print(dest.relative_to(ROOT))
