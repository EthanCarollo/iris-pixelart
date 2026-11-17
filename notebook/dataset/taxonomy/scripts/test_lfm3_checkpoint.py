import gzip
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import zlib

from PIL import Image

from annotate_lfm3 import open_database
from lfm3_schema import load_taxonomies
from sync_lfm3_checkpoint import prepare_snapshot


class CheckpointTests(unittest.TestCase):
    def test_snapshot_merges_packed_results_copies_crops_and_preserves_live_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'live'
            source.mkdir()
            crop = source / 'crops/parent/crop_00000.png'
            crop.parent.mkdir(parents=True)
            Image.new('RGBA', (8, 8), 'blue').save(crop)
            manifest = {'config_id': 'test', 'repo': 'example/private', 'revision': 'source-revision',
                        'model_revision': 'model-revision'}
            (source / 'run_manifest.json').write_text(json.dumps(manifest))
            (source / 'inventory.json').write_text(json.dumps({'paths': ['data/test.png']}))
            (source / 'ssh-password').write_text('do-not-copy-this')
            dimensions, leaves, _ = load_taxonomies()
            labels = {key: ['blue'] if dimensions[key].get('cardinality') == 'multi_label'
                      else next(iter(values)) for key, values in leaves.items()}
            labels.update(palette_size='limited', background_type='transparent',
                          background_scene='not_applicable')
            result = {'run_config_id': 'test', 'caption_candidate': 'A small blue sprite.',
                      'labels': labels, 'measurements': {'palette_size': 'limited', 'has_transparency': True},
                      'self_review': {'performed': True}, 'source': {'path': 'data/test.png'},
                      'execution_device': 'cpu', 'split': {'status': 'not_a_sheet'}}
            db = open_database(source / 'annotations.sqlite3')
            db.execute('INSERT INTO jobs(path,status,result) VALUES(?,?,?)',
                       ('data/test.png', 'done', zlib.compress(json.dumps(result).encode())))
            db.execute('INSERT INTO jobs(path,status) VALUES(?,?)', ('data/inflight.png', 'cpu_running'))
            db.execute('INSERT INTO jobs(path,local_path,parent_path,bbox,status) VALUES(?,?,?,?,?)',
                       ('derived/parent/crop_00000.png', str(crop), 'data/sheet.png', '[0,0,8,8]', 'pending'))
            db.commit()
            staging = root / 'snapshot'
            summary, assets = prepare_snapshot(source, staging)
            self.assertEqual(summary['annotations'], 1)
            self.assertEqual(summary['annotations_by_device'], {'cpu': 1})
            self.assertEqual(summary['derived_jobs'], 1)
            self.assertFalse(summary['complete'])
            self.assertEqual(db.execute('SELECT status FROM jobs WHERE path="data/inflight.png"').fetchone()[0], 'cpu_running')
            self.assertFalse((staging / 'ssh-password').exists())
            self.assertTrue((staging / 'crops/parent/crop_00000.png').exists())
            self.assertEqual(len(assets), 1)
            with gzip.open(staging / 'annotations_lfm3.jsonl.gz', 'rt') as stream:
                self.assertEqual(json.loads(stream.readline())['source']['dataset_path'], 'data/test.png')
            restored = root / 'restored.sqlite3'
            with gzip.open(staging / 'annotations.sqlite3.gz', 'rb') as stream:
                restored.write_bytes(stream.read())
            with sqlite3.connect(restored) as restored_db:
                self.assertEqual(restored_db.execute('SELECT status FROM jobs WHERE path="data/inflight.png"').fetchone()[0], 'pending')
                self.assertEqual(restored_db.execute('SELECT local_path FROM jobs WHERE parent_path IS NOT NULL').fetchone()[0], 'crops/parent/crop_00000.png')
            db.close()


if __name__ == '__main__':
    unittest.main()
