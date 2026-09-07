"""Create a fresh W&B project and bind only this module to it."""
import argparse
import datetime
import json
from pathlib import Path
import uuid

import wandb


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('configs/wandb.local.json'))
    p.add_argument('--entity',help='Defaults to the authenticated account default entity')
    args=p.parse_args()
    if args.output.exists():
        raise ValueError('Binding already exists; reuse it instead of creating another project')
    api=wandb.Api(timeout=20)
    entity=args.entity or api.default_entity
    if not entity:
        raise ValueError('No default entity; supply your W&B team with --entity')
    existing={p.name for p in api.projects(entity=entity)}
    name='PUBGAudio'
    if name in existing:
        name='PUBGAudio-'+datetime.date.today().strftime('%Y%m%d')+'-'+uuid.uuid4().hex[:6]
    api.create_project(name,entity=entity)
    project=api.project(name,entity=entity)
    binding=dict(application='PUBGAudio',created_for_this_module=True,entity=entity,
        project=name,project_id=project.id,url=project.url)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f:
        f.write(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(binding),flush=True)


if __name__=='__main__':
    main()
