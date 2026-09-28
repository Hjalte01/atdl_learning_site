"""Regression tests for live-vault selection, preservation, and failure reporting."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class MaterialsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'scripts').mkdir()
        (self.root / 'content/materials').mkdir(parents=True)
        (self.root / 'content/papers').mkdir()
        shutil.copy(Path(__file__).with_name('materials.py'), self.root / 'scripts/materials.py')
        self.catalog = self.root / 'content/materials/catalog.json'
        self.catalog.write_text('{"inventoryComplete": false, "materials": []}')
        self.vault = self.root / 'vault'
        self.vault.mkdir()
        (self.root / 'content/materials/source.json').write_text(json.dumps({'vault': str(self.vault)}))
        self.env = dict(os.environ)
        self.env.pop('ATDL_VAULT', None)

    def pdf(self, name, data=None):
        path = self.vault / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data or b'%PDF-1.4\n' + name.encode())
        return path

    def guide(self, name, topic, order=1, status='ready', kind='paper', pdf=None):
        path = self.root / f'content/papers/{name}/index.md'
        path.parent.mkdir(exist_ok=True)
        fields = dict(title=name, topic=f'topic-{topic}', order=order, status=status, materialKind=kind)
        if pdf:
            fields['pdf'] = pdf
        path.write_text('---\n' + '\n'.join(f'{k}: {json.dumps(v)}' for k, v in fields.items()) + '\n---\nUseful existing guide.\n')
        return path

    def run_cli(self, *args, code=0):
        result = subprocess.run(['python3', str(self.root / 'scripts/materials.py'), *args], text=True, capture_output=True, env=self.env)
        self.assertEqual(result.returncode, code, result.stdout + result.stderr)
        self.assertEqual(result.stderr, '')
        return json.loads(result.stdout)

    def data(self):
        return json.loads(self.catalog.read_text())

    def save(self, data):
        self.catalog.write_text(json.dumps(data))

    def test_live_next_discovers_topic_three_after_completed_catalog(self):
        self.pdf('topic 2/T2_1_Test.pdf')
        self.run_cli('sync')
        data = self.data()
        data['materials'][0]['status'] = 'ready'
        self.save(data)
        self.assertIsNone(self.run_cli('next')['next'])
        for order, title in enumerate(['DINOv3', 'V-JEPA 2.1', 'SAM 3', 'AnyUp', 'Med-SA', 'Unified View of PEFT'], 1):
            self.pdf(f'topic 3/T3_{order}_{title}.pdf')
        result = self.run_cli('next')
        self.assertEqual(result['freshness'], 'live')
        self.assertEqual(result['next']['sourcePath'], 'topic 3/T3_1_DINOv3.pdf')
        self.assertEqual(len(self.data()['materials']), 7)
        for m in self.data()['materials']:
            copy = self.root / 'public' / m['pdf'].lstrip('/')
            self.assertEqual(hashlib.sha256(copy.read_bytes()).hexdigest(), m['sha256'])

    def test_priority_and_classification(self):
        for name in ['topic 5/T5_2_Test.PDF', 'topic 5/T5_1_Test.pdf', 'topic 4/T4_1_Test.pdf',
                     'topic 3/extra reading material/ATDL_FM_Reading_Materials.pdf', 'topic 5/extra/additional.pdf',
                     'topic 4/lecture slides.pdf', 'misc.pdf', 'learning_site/excluded.pdf', '.obsidian/excluded.pdf']:
            self.pdf(name)
        self.assertEqual(self.run_cli('next')['next']['sourcePath'], 'topic 5/T5_1_Test.pdf')
        data = self.data()
        self.assertEqual(len(data['materials']), 7)
        for m in data['materials']:
            if m['kind'] == 'paper':
                m['status'] = 'ready'
        self.save(data)
        self.assertEqual(self.run_cli('next')['next']['kind'], 'overview')
        data = self.data()
        for m in data['materials']:
            if m['kind'] == 'overview':
                m['status'] = 'ready'
        self.save(data)
        self.assertEqual(self.run_cli('next')['next']['kind'], 'slides')

    def test_repeat_sync_preserves_manual_metadata_and_guides(self):
        self.pdf('topic 2/T2_1_Test.pdf')
        self.run_cli('import', str(self.vault))
        data = self.data()
        data['materials'][0].update(title='Manual title', topic=4, order=3, kind='extra', guide='/papers/custom')
        self.save(data)
        guide = self.guide('custom', 4)
        before = guide.read_bytes()
        self.assertIsNone(self.run_cli('next')['next'])
        self.assertEqual(self.data()['materials'], data['materials'])
        self.assertEqual(guide.read_bytes(), before)

    def test_nonpaper_guide_is_not_accidentally_linked_to_paper(self):
        self.guide('overview', 3, kind='overview')
        self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.assertIsNone(self.run_cli('next')['next']['guide'])

    def test_linked_queued_guide_has_source_and_ready_overrides_catalog(self):
        self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.guide('dino', 3, status='queued')
        result = self.run_cli('next')
        self.assertEqual(result['next']['file'], 'content/papers/dino/index.md')
        self.assertTrue(result['next']['pdf'])
        data = self.data()
        data['materials'][0]['reviewRequired'] = False
        self.save(data)
        self.guide('dino', 3)
        self.assertIsNone(self.run_cli('next')['next'])

    def test_changed_pdf_preserves_original_and_requires_review(self):
        source = self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.run_cli('sync')
        data = self.data()
        old = dict(data['materials'][0])
        data['materials'][0].update(guide='/papers/dino', status='ready')
        self.save(data)
        guide = self.guide('dino', 3, pdf=old['pdf'])
        old_guide = guide.read_bytes()
        original = source.read_bytes()
        source.write_bytes(b'%PDF-1.4\nnew revision')
        changed = self.run_cli('next')['next']
        self.assertTrue(changed['reviewRequired'])
        self.assertNotEqual(changed['pdf'], old['pdf'])
        self.assertEqual(changed['previousVersions'], [{'pdf': old['pdf'], 'sha256': old['sha256']}])
        self.assertEqual(guide.read_bytes(), old_guide)
        self.assertEqual((self.root / 'public' / old['pdf'].lstrip('/')).read_bytes(), original)
        # Re-importing or repairing the new copy cannot overwrite the guide's old PDF.
        (self.root / 'public' / changed['pdf'].lstrip('/')).unlink()
        self.assertEqual(self.run_cli('next')['next'], changed)
        self.assertEqual((self.root / 'public' / old['pdf'].lstrip('/')).read_bytes(), original)
        data = self.data()
        data['materials'][0]['reviewRequired'] = False
        self.save(data)
        self.assertIsNone(self.run_cli('next')['next'])

    def test_missing_vault_blocks_and_offline_is_explicit_and_read_only(self):
        self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.run_cli('sync')
        result = self.run_cli('next', '--vault', str(self.root / 'absent'), code=2)
        self.assertFalse(result['inventoryComplete'])
        self.assertIsNone(result['next'])
        self.assertIn('blocked', result)
        before = self.catalog.read_bytes()
        result = self.run_cli('next', '--offline')
        self.assertEqual(result['freshness'], 'cached')
        self.assertEqual(self.catalog.read_bytes(), before)
        self.assertEqual(result['next']['sourceState'], 'unverified')
        self.assertTrue(self.run_cli('next')['inventoryComplete'])

    def test_empty_vault_does_not_claim_complete(self):
        self.assertFalse(self.run_cli('next', code=2)['inventoryComplete'])

    @unittest.skipIf(os.geteuid() == 0, 'Filesystem permission fixture requires an unprivileged user')
    def test_unreadable_directory_blocks_lower_priority_and_preserves_records(self):
        self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.run_cli('sync')
        self.pdf('topic 2/extra/new.pdf')
        directory = self.vault / 'topic 3'
        directory.chmod(0)
        self.addCleanup(directory.chmod, 0o700)
        result = self.run_cli('next', code=2)
        self.assertIsNone(result['next'])
        self.assertIn('Permission denied', ' '.join(result['sync']['errors']))
        self.assertEqual(len(self.data()['materials']), 2)
        old = next(m for m in self.data()['materials'] if m['topic'] == 3)
        self.assertEqual(old['sourceState'], 'unverified')

    @unittest.skipIf(os.geteuid() == 0, 'Filesystem permission fixture requires an unprivileged user')
    def test_unreadable_pdf_blocks_with_partial_import(self):
        path = self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.pdf('topic 2/T2_1_Test.pdf')
        path.chmod(0)
        self.addCleanup(path.chmod, 0o600)
        result = self.run_cli('next', code=2)
        self.assertFalse(result['inventoryComplete'])
        self.assertEqual(result['sync']['importedCount'], 1)
        self.assertIsNone(result['next'])

    def test_disappeared_high_priority_source_is_retained_and_blocks(self):
        source = self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.pdf('topic 2/T2_1_Test.pdf')
        self.run_cli('sync')
        source.unlink()
        result = self.run_cli('next', code=2)
        self.assertEqual(result['next']['sourceState'], 'missing')
        self.assertEqual(len(self.data()['materials']), 2)
        self.assertIn('disappeared', result['blocked'])
        self.assertTrue((self.root / 'public' / result['next']['pdf'].lstrip('/')).exists())

    def test_vault_override_precedence(self):
        self.pdf('topic 3/T3_1_DINOv3.pdf')
        self.env['ATDL_VAULT'] = str(self.root / 'absent')
        self.run_cli('next', code=2)
        self.assertTrue(self.run_cli('next', '--vault', str(self.vault))['inventoryComplete'])

    def test_duplicate_pdf_records_are_preserved_when_source_disappears(self):
        self.pdf('topic 2/T2_1_Test.pdf')
        self.run_cli('sync')
        data = self.data()
        data['materials'].append(dict(data['materials'][0], id='manual-id', sourcePath='renamed.pdf'))
        self.save(data)
        self.run_cli('sync')
        self.assertEqual(len(self.data()['materials']), 2)
        self.assertEqual(self.data()['materials'][1]['sourceState'], 'missing')


if __name__ == '__main__':
    unittest.main(verbosity=2)
