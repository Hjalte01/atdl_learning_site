#!/usr/bin/env python3
"""Import vault PDFs or select the next guide. No third-party dependencies."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / 'content/materials/catalog.json'

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
    vault = vault.resolve(strict=True)
    def fail(error):
        raise error
    paths = []
    for directory, folders, filenames in os.walk(vault, onerror=fail):
        folders[:] = sorted(f for f in folders if f not in {'learning_site', 'atdl_learning_site', '.git', '.obsidian'})
        paths.extend(Path(directory) / f for f in filenames if Path(f).suffix.lower() == '.pdf')
    paths.sort()
    if not paths:
        raise SystemExit('No PDFs found; existing catalog preserved.')
    data = catalog()
    existing = {m['sourcePath']: m for m in data['materials']}
    papers = paper_records()
    imported = []
    for path in paths:
        source = path.relative_to(vault).as_posix()
        material_id = hashlib.sha256(source.encode()).hexdigest()[:16]
        match = re.search(r'(?:topic[ _-]*|\bT)([1-6])(?:[_ -]+(\d+))?', source, re.I)
        topic = int(match[1]) if match else None
        numbered = re.search(r'(?:topic[ _-]*|\bT)([1-6])[_ -]+(\d+)', path.stem, re.I)
        if numbered:
            topic = int(numbered[1])
        order = int(numbered[2]) if numbered else None
        lower = source.lower()
        kind = ('overview' if re.search(r'overview|reading[ _-]*list|syllabus', lower) else
                'slides' if re.search(r'slides?|lecture|presentation', lower) else
                'extra' if re.search(r'extra|supplement|additional', lower) else
                'paper' if order else 'extra')
        paper = next((p for p in papers if p['topic'] == f'topic-{topic}' and p['order'] == order), None) if kind == 'paper' else None
        old = existing.get(source, {})
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        # Reuse existing course PDFs only when their bytes match.
        known_pdf = paper.get('pdf') if paper else None
        known = ROOT / 'public' / known_pdf.lstrip('/') if known_pdf else None
        pdf = known_pdf if known and known.exists() and hashlib.sha256(known.read_bytes()).hexdigest() == digest else f'/materials/{material_id}.pdf'
        target = ROOT / 'public' / pdf.lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            shutil.copyfile(path, target)
        imported.append(dict(id=material_id, sourcePath=source, title=old.get('title', paper['title'] if paper else path.stem.replace('_', ' ')),
                             topic=old.get('topic', topic), order=old.get('order', order), kind=old.get('kind', kind),
                             status=old.get('status', 'not-started'), guide=old.get('guide', f"/papers/{paper['id']}" if paper else None),
                             pdf=pdf, sha256=digest))
    # Preserve prior records when a source disappears; never silently discard learning work.
    current_sources = {m['sourcePath'] for m in imported}
    imported += [m for m in data['materials'] if m['sourcePath'] not in current_sources and m['pdf'] not in {i['pdf'] for i in imported}]
    data.update(materials=imported, inventoryComplete=True)
    CATALOG.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(f'Imported {len(paths)} source PDFs. Review topic/kind/title inference in {CATALOG.relative_to(ROOT)}.')

def next_guide():
    data = catalog()
    papers = paper_records()
    candidates = [dict(id=p['id'], title=p['title'], topic=int(p['topic'].split('-')[1]), order=p['order'], kind=p.get('materialKind', 'paper'), file=p['file'])
                  for p in papers if p['status'] in ('queued', 'not-started')]
    for m in data['materials']:
        linked = next((p for p in papers if m.get('guide') == f"/papers/{p['id']}"), None)
        if not linked and m['status'] == 'not-started':
            candidates.append(m)
    candidates.sort(key=lambda m: ({'paper': 0, 'overview': 1, 'slides': 2, 'extra': 3}[m['kind']], -(m['topic'] or 0), m.get('order') or 999, m['id']))
    print(json.dumps({'inventoryComplete': data['inventoryComplete'], 'next': candidates[0] if candidates else None}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('import').add_argument('vault', type=Path)
    sub.add_parser('next')
    args = parser.parse_args()
    import_pdfs(args.vault) if args.command == 'import' else next_guide()
