"""Prepare and verify small records before authorized replacement of old synthetic data.

This tool never deletes files. Native PowerShell performs deletion only after the
new dataset gates and this archive have been verified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile


TARGETS = [
    'light-v5_3-20260907', 'light-v5_3-gunvehicle-20260907',
    'cache-light-v5_3-20260907', 'cache-light-v5_3-gunvehicle-20260907',
    'scenario-small-20260907', 'scenario-smoke-v4']
TEXT_SUFFIXES = {'.json', '.jsonl', '.py', '.md', '.txt', '.yaml', '.yml', '.toml', '.csv'}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def files_without_links(root):
    if root.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise ValueError(f'Reparse point: {root}')
    result = []
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in dirs+files:
            path = Path(directory)/name
            if path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError(f'Reparse point: {path}')
        result.extend(Path(directory)/name for name in files)
    return sorted(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    module = Path(__file__).resolve().parents[1]
    datasets = (module/'datasets').resolve(strict=True)
    if args.archive.exists() or args.receipt.exists():
        raise ValueError('Use fresh archive and receipt paths')
    inventory = {}
    for name in TARGETS:
        root = datasets/name
        if not root.is_dir():
            raise ValueError(f'Expected retired dataset missing: {root}')
        resolved = root.resolve(strict=True)
        if resolved.parent != datasets or resolved.name != name:
            raise ValueError(f'Unexpected resolved target: {resolved}')
        files = files_without_links(root)
        inventory[name] = dict(path=str(resolved), files=len(files), bytes=sum(f.stat().st_size for f in files),
            metadata_files=[dict(path=f.relative_to(datasets).as_posix(), sha256=digest(f))
                            for f in files if f.suffix in TEXT_SUFFIXES])
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in inventory.values():
            for f in item['metadata_files']:
                archive.write(datasets/f['path'], f['path'])
    with zipfile.ZipFile(args.archive) as archive:
        assert archive.testzip() is None
        for item in inventory.values():
            for f in item['metadata_files']:
                assert hashlib.sha256(archive.read(f['path'])).hexdigest() == f['sha256']
    retained = {str(f.relative_to(module)):dict(bytes=f.stat().st_size, sha256=digest(f))
                for f in (module/'outputs').glob('train-*/model*.pt')}
    result = dict(status='archive_prepared_and_verified_no_deletion', targets=inventory,
        total_bytes_to_delete=sum(x['bytes'] for x in inventory.values()),
        archive=str(args.archive.resolve()), archive_bytes=args.archive.stat().st_size,
        archive_sha256=digest(args.archive), retained_models=retained,
        scope='Only listed obsolete synthetic datasets and feature caches; originals, models and previews retained')
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['targets','retained_models']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
