"""Append training scenes while copying and freezing all parent recordings."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import generate_light_dataset as base


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sources',type=Path,default=Path('datasets/sources-v5_3-final'))
    p.add_argument('--hrir',type=Path,default=Path('datasets/hrir/kemar-diffuse.zip'))
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();root=args.output
    if args.workers<1:raise ValueError('Positive worker count required')
    spec=json.loads(args.spec.read_text(encoding='utf-8'))
    parent_cfg=json.loads((args.parent/'config.json').read_text())
    verified=json.loads((args.parent/'verification.json').read_text())
    if verified['status']!='passed':raise ValueError('Verify parent before extending it')
    parent_rows={s:[json.loads(line) for line in (args.parent/f'manifest_{s}.jsonl').read_text().splitlines()]
                 for s in base.SPLITS}
    if spec['source_groups_by_split']!=parent_cfg['light_release']['source_groups_by_split']:
        raise ValueError('Do not reassign held-out source families')
    if spec['clip_seconds']!=parent_cfg['clip_seconds']:raise ValueError('Incompatible parent duration')
    for s in ['validation','test']:
        if spec['split_counts'][s]!=len(parent_rows[s]):raise ValueError('Held-out scenes must remain fixed')
    added=spec['split_counts']['train']-len(parent_rows['train'])
    if added<1:raise ValueError('Specify additional training scenes')
    cfg=base.configuration(spec)
    cfg['purpose']='gunfire/vehicle-focused training extension; held-out data frozen'
    cfg['display_version']='v5.3-light-gunvehicle'
    manifest=json.loads((args.sources/'sources.json').read_text())
    pools=base.source_partitions(manifest['sources'],spec)
    targets=json.loads((args.parent/'training_targets.json').read_text())
    parent_hashes={s:base.digest(args.parent/f'manifest_{s}.jsonl') for s in base.SPLITS}
    if parent_hashes!=verified['manifests_sha256']:raise ValueError('Parent manifests changed')
    identity=dict(spec_sha256=base.digest(args.spec),source_manifest_sha256=base.digest(args.sources/'sources.json'),
        hrir_sha256=base.digest(args.hrir),targets=targets,parent_manifests_sha256=parent_hashes,
        generator_sha256={p.name:base.digest(p) for p in Path(__file__).parent.glob('*.py')})
    if root.exists():
        if not args.resume:raise ValueError('Preserve existing data; use a new directory or --resume')
        if json.loads((root/'generation_identity.json').read_text())!=identity:
            raise ValueError('Configuration or generator changed since extension began')
    for folder in ['audio','recipes','rows','generator_snapshot','parent_generator_snapshot','parent_manifests']:
        (root/folder).mkdir(parents=True,exist_ok=True)
    base.write_json(root/'generation_identity.json',identity)
    base.write_json(root/'config.json',cfg)
    base.write_json(root/'parent_config.json',parent_cfg)
    base.write_json(root/'training_targets.json',targets)
    base.write_json(root/'source_manifest.json',manifest)
    shutil.copy2(args.parent/'source_split.json',root/'source_split.json')
    shutil.copy2(args.parent/'generation_identity.json',root/'parent_generation_identity.json')
    for path in (args.parent/'generator_snapshot').glob('*.py'):
        shutil.copy2(path,root/'parent_generator_snapshot'/path.name)
    for path in Path(__file__).parent.glob('*.py'):
        shutil.copy2(path,root/'generator_snapshot'/path.name)
    inherited=[]
    for split,rows in parent_rows.items():
        shutil.copy2(args.parent/f'manifest_{split}.jsonl',root/'parent_manifests'/f'manifest_{split}.jsonl')
        for row in rows:
            inherited.append(row['id'])
            for field,key in [('audio_file','audio_sha256'),('label_file','label_sha256'),
                              ('detailed_label_file','detailed_label_sha256'),('recipe_file','recipe_sha256')]:
                source=args.parent/row[field];destination=root/row[field]
                if base.digest(source)!=row[key]:raise ValueError('Parent file changed: '+str(source))
                if not destination.exists():shutil.copy2(source,destination)
                if base.digest(destination)!=row[key]:raise ValueError('Copied parent differs: '+str(destination))
            for suffix in ['.json','.stats.json']:
                shutil.copy2(args.parent/'rows'/(row['id']+suffix),root/'rows'/(row['id']+suffix))
    for pool in pools.values():
        for source in pool:
            if base.digest(args.sources/source['wav_file'])!=source['wav_sha256']:
                raise ValueError('Source asset changed')
    offset=len(parent_rows['train'])
    jobs=[('train',offset+i+1,spec['seed']+i) for i in range(added)]
    if {row['seed'] for rows in parent_rows.values() for row in rows}&{j[2] for j in jobs}:
        raise ValueError('Use fresh scene seeds')
    new_ids=[f'train_{j[1]:04d}' for j in jobs]
    extension=dict(parent=str(args.parent.resolve()),parent_manifests_sha256=parent_hashes,
        inherited_scene_ids=inherited,new_scene_ids=new_ids,added_seconds=added*cfg['clip_seconds'],
        probability_multipliers=spec['spawn_probability_multipliers'],held_out_copied_byte_identically=True)
    base.write_json(root/'extension.json',extension)
    print(json.dumps(dict(status='extending_training_only',added_scenes=added,added_seconds=extension['added_seconds'],
                          final_split_counts=spec['split_counts'])),flush=True)
    pending=[job for job in jobs if not (root/'rows'/f'train_{job[1]:04d}.stats.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=base.initialize,
            initargs=(root,args.sources,args.hrir,cfg,pools,targets)) as executor:
        for result in executor.map(base.generate,pending):
            print(json.dumps(result),flush=True)
    new_rows=[json.loads((root/'rows'/(sid+'.json')).read_text()) for sid in new_ids]
    for split,rows in parent_rows.items():
        parent_text=(root/'parent_manifests'/f'manifest_{split}.jsonl').read_text(encoding='utf-8')
        if split=='train':
            parent_text+=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in new_rows)
            (root/f'manifest_{split}.jsonl').write_text(parent_text,encoding='utf-8')
        else:
            shutil.copy2(root/'parent_manifests'/f'manifest_{split}.jsonl',root/f'manifest_{split}.jsonl')
            assert base.digest(root/f'manifest_{split}.jsonl')==parent_hashes[split]
    base.write_plan(root, targets)
    rows=parent_rows['train']+new_rows+parent_rows['validation']+parent_rows['test']
    stats=[json.loads((root/'rows'/(row['id']+'.stats.json')).read_text()) for row in rows]
    with (root/'recipes.jsonl').open('w',encoding='utf-8') as output:
        for row in rows:output.write((root/row['recipe_file']).read_text(encoding='utf-8'))
    base.write_json(root/'summary.json',dict(status='rendered_pending_verification',clips=len(rows),
        seconds=sum(row['duration_seconds'] for row in rows),split_counts=spec['split_counts'],
        results=stats,source_manifest_sha256=identity['source_manifest_sha256'],hrir_sha256=identity['hrir_sha256'],
        split_density={s:dict(Counter(row['density'] for row in rows if row['split']==s)) for s in base.SPLITS},
        extension=extension,limitations=spec['notes']))
    print(json.dumps(dict(status='extension_complete',clips=len(rows),new_scenes=added)),flush=True)


if __name__=='__main__':main()
