"""Copy verified local source libraries into one module-local dataset directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=args.output.resolve(); root.mkdir(parents=True,exist_ok=True)
    sources={}; seen={}
    for folder in args.input:
        manifest=json.loads((folder/'sources.json').read_text(encoding='utf-8'))
        for s in manifest['sources']:
            for key,hashkey in [('wav_file','wav_sha256'),('wem_file',None)]:
                rel=Path(s[key]); src=(folder/rel).resolve(); dest=(root/rel).resolve()
                if not src.is_relative_to(folder.resolve()) or not dest.is_relative_to(root):
                    raise ValueError('Source path escapes dataset')
                digest=hashlib.sha256(src.read_bytes()).hexdigest()
                expected=s[hashkey] if hashkey else s['provenance']['wem_sha256']
                if digest!=expected:
                    raise ValueError('Source hash mismatch '+s['id'])
                if dest.exists():
                    if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:
                        raise ValueError('Refuse changed original '+str(dest))
                else:
                    dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dest)
            if s['id'] in sources and sources[s['id']]!=s:
                raise ValueError('Conflicting source identity')
            if s['pcm_sha256'] in seen and seen[s['pcm_sha256']]!=s['id']:
                raise ValueError('Duplicate PCM under different source identities')
            sources[s['id']]=s; seen[s['pcm_sha256']]=s['id']
    result=dict(schema_version=1,source_policy='verified_named_local_client_media_read_only',
        source_count=len(sources),sources=list(sources.values()),excluded=[],duplicates=[])
    (root/'sources.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(sources=len(sources),output=str(root))))


if __name__=='__main__':
    main()
