"""Exercise imports and priority decisions in an isolated fixture repository."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    (root / 'scripts').mkdir()
    (root / 'content/materials').mkdir(parents=True)
    (root / 'content/papers').mkdir()
    shutil.copy(Path(__file__).with_name('materials.py'), root / 'scripts/materials.py')
    catalog = root / 'content/materials/catalog.json'
    catalog.write_text('{"inventoryComplete": false, "materials": []}')
    vault = root / 'vault'
    for name in ['Topic 5/T5_2_Test.PDF', 'Topic 5/overview.pdf', 'Topic 4/lecture slides.pdf', 'Topic 5/extra/additional.pdf', 'misc.pdf']:
        path = vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'%PDF-1.4\n' + name.encode())
    def run(*args):
        return subprocess.check_output(['python3', str(root / 'scripts/materials.py'), *args], text=True)
    run('import', str(vault))
    data = json.loads(catalog.read_text())
    assert len(data['materials']) == 5
    for material in data['materials']:
        assert hashlib.sha256((root / 'public' / material['pdf'].lstrip('/')).read_bytes()).hexdigest() == material['sha256']
    paper = next(m for m in data['materials'] if m['kind'] == 'paper')
    assert paper['topic'] == 5 and paper['order'] == 2
    assert json.loads(run('next'))['next']['id'] == paper['id']
    paper.update(status='ready', title='Manually corrected title')
    catalog.write_text(json.dumps(data))
    run('import', str(vault))
    data = json.loads(catalog.read_text())
    assert len(data['materials']) == 5
    assert next(m for m in data['materials'] if m['id'] == paper['id'])['title'] == 'Manually corrected title'
    assert json.loads(run('next'))['next']['kind'] == 'overview'
    for topic in (2, 4):
        path = root / f'content/papers/t{topic}-paper-1/index.md'
        path.parent.mkdir()
        path.write_text(f'---\ntitle: "Paper"\ntopic: "topic-{topic}"\norder: 1\nstatus: "queued"\n---\n')
    assert json.loads(run('next'))['next']['topic'] == 4
    # A linked ready guide overrides a stale catalog status and is never selected twice.
    overview = next(m for m in data['materials'] if m['kind'] == 'overview')
    overview['guide'] = '/papers/overview'
    catalog.write_text(json.dumps(data))
    path = root / 'content/papers/overview/index.md'
    path.parent.mkdir()
    path.write_text('---\ntitle: "Overview"\ntopic: "topic-5"\norder: 1\nstatus: "ready"\nmaterialKind: "overview"\n---\n')
    for path in (root / 'content/papers').glob('t*/index.md'):
        path.write_text(path.read_text().replace('"queued"', '"ready"'))
    assert json.loads(run('next'))['next']['kind'] == 'slides'
print('PASS: all PDFs imported with matching hashes; nested numbering, repeat import, metadata preservation, category/topic priority, and linked readiness.')
