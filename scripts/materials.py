#!/usr/bin/env python3
"""Sync the live study vault before selecting a guide. No third-party dependencies."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'content/materials/catalog.json'
SOURCE = ROOT / 'content/materials/source.json'


def source_vault(override=None):
    if override:
        return override
    if os.environ.get('ATDL_VAULT'):
        return Path(os.environ['ATDL_VAULT'])
    return Path(json.loads(SOURCE.read_text())['vault'])


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_catalog(data):
    # A failed/interrupted write must not truncate the inventory.
    with tempfile.NamedTemporaryFile(mode='w', dir=CATALOG.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    temporary.replace(CATALOG)


def paper_records():
    records = []
    for path in sorted((ROOT / 'content/papers').glob('*/index.md')):
        frontmatter = path.read_text().split('---', 2)[1]
        data = {}
        for key, value in re.findall(r'^(\w+):\s*(.+)$', frontmatter, re.M):
            try:
                data[key] = json.loads(value)
            except ValueError:
                data[key] = value.strip('\"\'')
        records.append(dict(data, id=path.parent.name, file=str(path.relative_to(ROOT))))
    return records


def catalog():
    return json.loads(CATALOG.read_text())


def import_pdfs(vault):
    vault = vault.expanduser().absolute()
    errors = []

    def fail(error):
        errors.append(str(error))

    paths = []
    for directory, folders, filenames in os.walk(vault, onerror=fail):
        folders[:] = sorted(f for f in folders if f not in {'learning_site', 'atdl_learning_site', '.git', '.obsidian'})
        paths.extend(Path(directory) / f for f in filenames if Path(f).suffix.lower() == '.pdf')
    paths.sort()
    if not paths and not errors:
        errors.append(f'No PDFs found in {vault}; check the vault path or mount.')
    data = catalog()
    existing = {m['sourcePath']: m for m in data['materials']}
    papers = paper_records()
    imported = []
    for path in paths:
        source = path.relative_to(vault).as_posix()
        old = existing.get(source, {})
        material_id = old.get('id', hashlib.sha256(source.encode()).hexdigest()[:16])
        match = re.search(r'(?:topic[ _-]*|\bT)([1-6])(?:[_ -]+(\d+))?', source, re.I)
        topic = int(match[1]) if match else None
        numbered = re.search(r'(?:topic[ _-]*|\bT)([1-6])[_ -]+(\d+)', path.stem, re.I)
        if numbered:
            topic = int(numbered[1])
        order = int(numbered[2]) if numbered else None
        lower = source.lower()
        kind = ('overview' if re.search(r'overview|reading[ _-]*(?:list|materials)|syllabus', path.stem.lower()) or 'overview/' in lower else
                'slides' if re.search(r'slides?|lecture|presentation', lower) else
                'extra' if re.search(r'extra|supplement|additional', lower) else
                'paper' if order else 'extra')
        paper = next((p for p in papers if p.get('materialKind', 'paper') == 'paper' and p['topic'] == f'topic-{topic}' and p['order'] == order), None) if kind == 'paper' else None
        try:
            digest = sha256(path)
            changed = bool(old and old['sha256'] != digest)
            # Changed sources get a new asset; existing guides keep their original PDF.
            known_pdf = paper.get('pdf') if paper else None
            known = ROOT / 'public' / known_pdf.lstrip('/') if known_pdf else None
            known_matches = bool(known and known.exists() and sha256(known) == digest)
            pdf = old['pdf'] if old and not changed else known_pdf if known_matches else f'/materials/{material_id}{"-" + digest if changed else ""}.pdf'
            target = ROOT / 'public' / pdf.lstrip('/')
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or sha256(target) != digest:
                with tempfile.TemporaryDirectory(dir=target.parent) as staging:
                    copy = Path(staging) / 'source.pdf'
                    shutil.copyfile(path, copy)
                    if sha256(copy) != digest:
                        raise OSError(f'Source changed during import: {path}; retry sync.')
                    copy.replace(target)
        except OSError as error:
            fail(error)
            continue
        material = dict(old, id=material_id, sourcePath=source, title=old.get('title', paper['title'] if paper else path.stem.replace('_', ' ')),
                        topic=old.get('topic', topic), order=old.get('order', order), kind=old.get('kind', kind),
                        status=old.get('status', 'not-started'), guide=old.get('guide', f"/papers/{paper['id']}" if paper else None),
                        pdf=pdf, sha256=digest, sourceState='present')
        if changed:
            material['previousVersions'] = [*old.get('previousVersions', []), {'pdf': old['pdf'], 'sha256': old['sha256']}]
            material['reviewRequired'] = bool(material['guide'] or material['status'] == 'ready')
        if not old and paper and paper['status'] == 'ready' and not known_matches:
            material['reviewRequired'] = True
        imported.append(material)
    # Preserve every prior record, including duplicate PDF references and manual metadata.
    current_sources = {m['sourcePath'] for m in imported}
    imported += [dict(m, sourceState='unverified' if errors else 'missing') for m in data['materials'] if m['sourcePath'] not in current_sources]
    now = datetime.now(timezone.utc).isoformat(timespec='seconds')
    sync = dict(data.get('sync', {}), vault=str(vault), attemptedAt=now,
                sourceCount=len(paths), importedCount=len(current_sources), errors=errors)
    if not errors:
        sync['completedAt'] = now
    data.update(materials=imported, inventoryComplete=not errors, sync=sync)
    save_catalog(data)
    return data


def next_guide(data, offline=False):
    result = {'inventoryComplete': data['inventoryComplete'], 'freshness': 'cached' if offline else 'live',
              'sync': data.get('sync'), 'next': None}
    if not offline and not data['inventoryComplete']:
        result['blocked'] = 'Vault sync is incomplete; restore source access and retry. No lower-priority guide selected.'
        return result
    papers = paper_records()
    candidates = [dict(id=p['id'], title=p['title'], topic=int(p['topic'].split('-')[1]), order=p['order'], kind=p.get('materialKind', 'paper'), file=p['file'])
                  for p in papers if p['status'] in ('queued', 'not-started')]
    for m in data['materials']:
        linked = next((p for p in papers if m.get('guide') == f"/papers/{p['id']}"), None)
        if linked:
            candidates = [c for c in candidates if c['id'] != linked['id']]
        if m.get('reviewRequired') or (linked['status'] in ('queued', 'not-started') if linked else m['status'] == 'not-started'):
            candidates.append(dict(m, **({'file': linked['file']} if linked else {})))
    candidates.sort(key=lambda m: ({'paper': 0, 'overview': 1, 'slides': 2, 'extra': 3}[m['kind']], -(m['topic'] or 0), m.get('order') or 999, m['id']))
    if candidates:
        selected = candidates[0]
        result['next'] = selected
        if not offline:
            if selected.get('sourceState') == 'missing':
                result['blocked'] = f'Selected source disappeared from the vault: {selected["sourcePath"]}. Restore it or explicitly resolve this record.'
            elif selected.get('pdf'):
                try:
                    if sha256(ROOT / 'public' / selected['pdf'].lstrip('/')) != selected['sha256']:
                        raise OSError('Imported PDF hash mismatch')
                except OSError as error:
                    result['blocked'] = f'Selected PDF unavailable: {error}'
            else:
                result['blocked'] = 'Selected guide has no catalog source; supply and link its PDF before authoring.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('import', help='Sync a specific vault (legacy command)').add_argument('vault', type=Path)
    sub.add_parser('sync', help='Refresh from the configured live vault').add_argument('--vault', type=Path)
    select = sub.add_parser('next', help='Sync the live vault, then select one guide')
    mode = select.add_mutually_exclusive_group()
    mode.add_argument('--vault', type=Path)
    mode.add_argument('--offline', action='store_true', help='Inspect cached selection only; does not verify current vault coverage')
    args = parser.parse_args()
    try:
        offline = getattr(args, 'offline', False)
        data = catalog() if offline else import_pdfs(source_vault(getattr(args, 'vault', None)))
        result = next_guide(data, offline) if args.command == 'next' else {'inventoryComplete': data['inventoryComplete'], 'sync': data['sync']}
        print(json.dumps(result, indent=2, ensure_ascii=False))
        sys.exit(2 if result.get('blocked') or (not offline and not data['inventoryComplete']) else 0)
    except (OSError, ValueError) as error:
        print(json.dumps({'inventoryComplete': False, 'next': None, 'blocked': str(error)}))
        sys.exit(2)
