"""Build a new local source pool without modifying existing pools."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('pools',nargs='+',type=Path);a=p.parse_args()
    if a.output.exists():raise ValueError('Choose a new output directory')
    sources={};inputs=[]
    for pool in a.pools:
        path=pool/'sources.json';manifest=json.loads(path.read_text(encoding='utf-8'))
        inputs.append(dict(path=str(pool),manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for s in manifest['sources']:
            if s['id'] in sources:
                if sources[s['id']][0]!=s:raise ValueError('Source identity conflict: '+s['id'])
                continue
            sources[s['id']]=(s,pool)
    a.output.mkdir(parents=True)
    for s,pool in sources.values():
        for key in ['wav_file','wem_file']:
            src=pool/s[key];target=a.output/s[key]
            target.parent.mkdir(exist_ok=True,parents=True)
            if key=='wav_file' and hashlib.sha256(src.read_bytes()).hexdigest()!=s['wav_sha256']:
                raise ValueError('Source WAV hash mismatch: '+s['id'])
            # Original source files are immutable; hard links avoid duplicate local storage.
            os.link(src,target)
    out=dict(schema_version=1,source_policy='named_local_client_media_read_only',inputs=inputs,
             source_count=len(sources),sources=[s for s,_ in sources.values()],
             limitation='Native source identity does not prove exact in-game timing, level, or fine-pose acoustic identity.')
    (a.output/'sources.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('MERGED',len(sources))


if __name__=='__main__':main()
