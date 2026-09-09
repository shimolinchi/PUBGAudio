"""Inventory local Wwise metadata and prepared WAVs; does not extract game files."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(metadata, sources):
    info = json.loads(metadata.read_text(encoding='utf-8').rstrip('\0'))['SoundBanksInfo']
    prepared = json.loads((sources / 'sources.json').read_text(encoding='utf-8'))['sources']
    assets = {}

    def register(media, location, bank=None):
        mid = str(media['Id'])
        item = assets.setdefault(mid, dict(media_id=mid, names=set(), locations=set(), banks=set(), prepared=[]))
        if media.get('ShortName'):
            item['names'].add(media['ShortName'])
        item['locations'].add(location)
        if bank:
            item['banks'].add(bank)

    for media in info['StreamedFiles']:
        register(media, 'streamed')
    banks = []
    for bank in info['SoundBanks']:
        embedded = bank.get('IncludedMemoryFiles', [])
        streamed = bank.get('ReferencedStreamedFiles', [])
        for media in embedded:
            register(media, 'embedded', bank['ShortName'])
        for media in streamed:
            register(media, 'streamed_reference', bank['ShortName'])
        banks.append(dict(name=bank['ShortName'], bank_id=bank['Id'],
                          embedded_references=len(embedded), streamed_references=len(streamed),
                          unique_media_count=len({str(m['Id']) for m in embedded + streamed})))
    errors = []
    for source in prepared:
        wav = sources / source['wav_file']
        if not wav.is_file():
            errors.append(dict(id=source['id'], error='missing WAV'))
        elif sha256(wav) != source['wav_sha256']:
            errors.append(dict(id=source['id'], error='WAV hash mismatch'))
        mid = str(source['media_id'])
        if mid not in assets:
            errors.append(dict(id=source['id'], error='missing metadata ID'))
        else:
            assets[mid]['prepared'].append(source['id'])
    for item in assets.values():
        for key in ('names', 'locations', 'banks'):
            item[key] = sorted(item[key])
    groups = defaultdict(Counter)
    for source in prepared:
        groups[source['source_group_id']][source['role']] += 1
    report = dict(
        schema_version=1,
        scope='This local metadata snapshot; media IDs are not independent acoustic classes or PCM hashes.',
        metadata_sha256=sha256(metadata), source_manifest_sha256=sha256(sources / 'sources.json'),
        bank_count=len(banks), embedded_reference_count=sum(b['embedded_references'] for b in banks),
        stream_directory_count=len(info['StreamedFiles']), unique_media_ids=len(assets),
        unique_wav_ids=sum(any(n.lower().endswith('.wav') for n in a['names']) for a in assets.values()),
        unique_model_ids=sum(any(n.lower().endswith('.model') for n in a['names']) for a in assets.values()),
        prepared_wavs=len(prepared), prepared_pcm_hashes=len({s['pcm_sha256'] for s in prepared}),
        prepared_roles=dict(sorted(Counter(s['role'] for s in prepared).items())),
        prepared_groups={key: dict(sorted(value.items())) for key, value in sorted(groups.items())},
        verification_errors=errors, banks=sorted(banks, key=lambda b: b['name']),
        assets=sorted(assets.values(), key=lambda a: int(a['media_id'])),
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--sources', type=Path, default=Path('datasets/sources-v4'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.metadata, args.sources)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'catalog.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 本地声音目录计数', '',
             '按媒体 ID 去重；不同 ID 仍可能有相同 PCM。目录存在不表示已解码、可播放或适用于普通对局。', '',
             '| 素材家族 | 已准备 WAV | 角色数量 |', '|---|---:|---|']
    for group, roles in result['prepared_groups'].items():
        lines.append(f"| {group} | {sum(roles.values())} | " + ', '.join(f'{k}={v}' for k, v in roles.items()) + ' |')
    lines += ['', '## 声音库完整列表', '',
              'bank 是客户端资源组织单位，不直接用作训练类别。一个媒体可被多个 bank 引用。', '',
              '| Bank | 内嵌引用 | 流式引用 | Bank 内唯一媒体 ID |', '|---|---:|---:|---:|']
    for b in result['banks']:
        lines.append(f"| {b['name']} | {b['embedded_references']} | {b['streamed_references']} | {b['unique_media_count']} |")
    (args.output / 'inventory_counts.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('assets', 'banks', 'prepared_groups')}, ensure_ascii=False))
    if result['verification_errors']:
        raise SystemExit('Prepared source verification failed; see catalog.json')


if __name__ == '__main__':
    main()
