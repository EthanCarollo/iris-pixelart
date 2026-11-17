"""Invariant checks for taxonomy constraints, source pixels, and lossless sheet splitting."""
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
import zlib
from unittest.mock import patch
from types import SimpleNamespace

from PIL import Image, ImageDraw

from annotate_lfm3 import model_image, selected_images
from lfm3_schema import load_taxonomies, json_schema, grammar, validate
from split_sprite_sheets import propose_crops


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.dimensions, self.leaves, _ = load_taxonomies()
        labels = {key: (['unknown'] if self.dimensions[key].get('cardinality') == 'multi_label'
                        else next(iter(values))) for key, values in self.leaves.items()}
        labels.update(background_type='transparent', background_scene='not_applicable', palette_size='limited')
        self.example = {'labels': labels, 'caption_candidate': 'A small isolated pixel-art shape.'}

    def test_validator_rejects_impossible_labels_and_mixed_unknown(self):
        validate(self.example, self.dimensions, self.leaves, 'limited', True)
        for patch in [dict(subject_type='invented'), dict(dominant_color_families=['red', 'red']),
                      dict(dominant_color_families=['red', 'unknown']), dict(background_scene='forest'),
                      dict(palette_size='very_broad'), dict(background_type='solid_color')]:
            bad = copy.deepcopy(self.example)
            bad['labels'].update(patch)
            with self.assertRaises(ValueError):
                validate(bad, self.dimensions, self.leaves, 'limited', True)

    def test_runtime_schema_has_exact_measurements_and_background_dependency(self):
        schema = json_schema(self.dimensions, self.leaves, 'limited', True)
        self.assertEqual(next(iter(schema['properties'])), 'caption_candidate')
        self.assertEqual(schema['properties']['caption_candidate']['pattern'], r'^A(n)? [^\n\r]{1,636}[.!?]$')
        fields = schema['properties']['labels']['properties']
        self.assertEqual(fields['palette_size']['enum'], ['limited'])
        self.assertEqual(fields['background_scene']['enum'], ['not_applicable'])
        branches = json_schema(self.dimensions, self.leaves, 'limited', False)['properties']['labels']['oneOf']
        for branch in branches:
            props = branch['properties']
            if props['background_type']['enum'] == ['solid_color']:
                self.assertEqual(props['background_scene']['enum'], ['not_applicable'])
            else:
                self.assertNotIn('not_applicable', props['background_scene']['enum'])
        self.assertIn('dominant-color-families-subset-0', grammar(self.dimensions, self.leaves))

    def test_caption_rejects_framing_and_incomplete_sentences(self):
        for caption in ['The image displays a blue creature.', 'A blue creature with a long']:
            result = copy.deepcopy(self.example)
            result['caption_candidate'] = caption
            with self.assertRaises(ValueError):
                validate(result, self.dimensions, self.leaves, 'limited', True)

    def test_alpha_zero_colors_are_not_counted_and_nearest_neighbor_keeps_clusters(self):
        image = Image.new('RGBA', (8, 8), (255, 0, 0, 0))
        image.putpixel((2, 2), (0, 255, 0, 255))
        image.putpixel((3, 2), (0, 0, 255, 128))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'image.png'
            image.save(path)
            rendered, measured = model_image(path)
        self.assertEqual(measured['visible_rgb_color_count'], 2)
        self.assertEqual(rendered.size, (512, 512))
        self.assertEqual(measured['width'], 8)
        self.assertTrue(measured['has_transparency'])

    def test_sheet_split_preserves_all_foreground_including_small_fragments(self):
        image = Image.new('RGBA', (48, 24))
        draw = ImageDraw.Draw(image)
        draw.rectangle((2, 3, 12, 18), fill='red')
        draw.rectangle((28, 4, 42, 20), fill='blue')
        boxes, status = propose_crops(image)
        self.assertEqual(boxes, [[2, 3, 13, 19], [28, 4, 43, 21]])
        self.assertEqual(status['foreground_coverage'], 1.0)
        image.putpixel((46, 3), (255, 255, 0, 255))
        boxes, _ = propose_crops(image)
        self.assertEqual(len(boxes), 3)

    def test_ambiguous_or_single_images_remain_explicitly_pending(self):
        image = Image.new('RGBA', (16, 16), 'red')
        boxes, status = propose_crops(image)
        self.assertFalse(boxes)
        self.assertEqual(status['status'], 'pending_geometry')

    def test_grid_does_not_discard_visible_pixels_in_gutters(self):
        image = Image.new('RGBA', (18, 8))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 7, 7), fill='red')
        draw.rectangle((10, 0, 17, 7), fill='blue')
        boxes, _ = propose_crops(image, {'cell_width': 8, 'cell_height': 8, 'spacing_x': 2})
        self.assertEqual(len(boxes), 2)
        image.putpixel((8, 2), (0, 255, 0, 255))
        with self.assertRaises(ValueError):
            propose_crops(image, {'cell_width': 8, 'cell_height': 8, 'spacing_x': 2})

    def test_all_selection_has_no_quality_or_annotation_filter(self):
        paths = ['data/one.png', 'data/two.gif', 'source_sheets/sheet.bmp']
        self.assertEqual(selected_images(paths, 0, 1, True), paths)

    def test_vllm_reviews_same_image_and_keeps_draft_and_actual_changes(self):
        from lfm3_vllm import reviewed_annotation, decoding_schema
        draft = copy.deepcopy(self.example)
        revised = copy.deepcopy(draft)
        revised['caption_candidate'] = 'A small blue pixel-art shape on a transparent background.'
        revised.update(review_decision='corrected', review_notes=['Clarified the visible color.'],
                       visual_checks={'instance_layout': 'single_subject'})
        response = {'usage': {'completion_tokens': 20}}
        measured = {'width': 16, 'height': 16, 'palette_size': 'limited',
                    'has_transparency': True, 'visible_rgb_color_count': 2}
        proposal = {**draft, 'visual_checks': {'instance_layout': 'single_subject'}}
        with patch('lfm3_vllm.chat', side_effect=[(proposal, response), (revised, response)]) as mock:
            final, audit, schema, _ = reviewed_annotation(
                'http://localhost', Image.new('RGB', (16, 16), 'blue'), measured,
                self.dimensions, self.leaves, 'Visual taxonomy instructions')
        self.assertEqual(mock.call_count, 2)
        first_messages, second_messages = mock.call_args_list[0].args[1], mock.call_args_list[1].args[1]
        self.assertEqual(first_messages[1]['content'][0], second_messages[1]['content'][0])
        self.assertIn(draft['caption_candidate'], second_messages[1]['content'][1]['text'])
        self.assertTrue(audit['changed'])
        self.assertEqual(audit['draft'], draft)
        self.assertEqual(final['caption_candidate'], revised['caption_candidate'])
        self.assertTrue(schema['properties']['labels']['properties']['dominant_color_families']['uniqueItems'])
        self.assertNotIn('uniqueItems', decoding_schema(schema)['properties']['labels']['properties']['dominant_color_families'])

    def test_layout_grammar_excludes_sheets_for_one_subject(self):
        from lfm3_vllm import constrain_layout
        original = json_schema(self.dimensions, self.leaves, 'limited', True)
        restricted = constrain_layout(original, 'single_subject')
        values = restricted['properties']['labels']['properties']['asset_type']['enum']
        self.assertIn('single_sprite', values)
        self.assertNotIn('sprite_sheet', values)
        self.assertNotIn('tile_set', values)
        self.assertIn('sprite_sheet', original['properties']['labels']['properties']['asset_type']['enum'])

    def test_duplicate_colors_are_normalized_as_a_set_with_an_audit(self):
        from lfm3_vllm import normalize_color_set
        result = copy.deepcopy(self.example)
        result['labels']['dominant_color_families'] = ['blue', 'red', 'blue']
        audit = []
        normalize_color_set(result, 'draft', audit)
        self.assertEqual(result['labels']['dominant_color_families'], ['blue', 'red'])
        self.assertEqual(audit[0]['before'], ['blue', 'red', 'blue'])
        validate(result, self.dimensions, self.leaves, 'limited', True)

    def test_source_sheet_is_split_even_if_model_calls_it_single_sprite(self):
        from annotate_lfm3 import open_database, schedule_splits
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / 'sheet.png'
            image = Image.new('RGBA', (48, 24))
            draw = ImageDraw.Draw(image)
            draw.rectangle((2, 3, 12, 18), fill='red')
            draw.rectangle((28, 4, 42, 20), fill='blue')
            image.save(path)
            db = open_database(output / 'jobs.sqlite3')
            result = {'source': {'path': 'source_sheets/test.png', 'parent_path': None},
                      'asset_id': 'sheet1', 'labels': {'asset_type': 'single_sprite'}}
            schedule_splits(db, result, path, SimpleNamespace(output=output, existing_splits={}), {})
            self.assertEqual(result['split']['crop_count'], 2)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM jobs').fetchone()[0], 2)
            self.assertEqual(db.execute('SELECT parent_path FROM jobs LIMIT 1').fetchone()[0], 'source_sheets/test.png')
            db.close()

    def test_compressed_database_exports_lossless_jsonl(self):
        from annotate_lfm3 import open_database, export, unpack_record
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = open_database(root / 'jobs.sqlite3')
            serialized = json.dumps(self.example, ensure_ascii=False)
            packed = zlib.compress(serialized.encode())
            db.execute('INSERT INTO jobs(path,status,result) VALUES(?,?,?)', ('data/test.png', 'done', packed))
            db.commit()
            self.assertEqual(json.loads(unpack_record(packed)), self.example)
            export(db, root, compress=True)
            with gzip.open(root / 'annotations_lfm3.jsonl.gz', 'rt') as stream:
                self.assertEqual(json.loads(stream.readline()), self.example)
            db.close()

    def test_auxiliary_cpu_claim_uses_tail_and_distinct_running_state(self):
        from annotate_lfm3 import open_database
        from run_lfm3_cpu_worker import claim_tail
        with tempfile.TemporaryDirectory() as directory:
            db = open_database(Path(directory) / 'queue.sqlite3')
            db.executemany('INSERT INTO jobs(path) VALUES(?)', [(f'data/{i}.png',) for i in range(5)])
            db.commit()
            job = claim_tail(db, min_pending=3)
            self.assertEqual(job[0], 'data/4.png')
            self.assertEqual(db.execute('SELECT status FROM jobs WHERE path=?', (job[0],)).fetchone()[0], 'cpu_running')
            self.assertEqual(db.execute('SELECT path FROM jobs WHERE status="pending" ORDER BY rowid LIMIT 1').fetchone()[0], 'data/0.png')
            self.assertIsNone(claim_tail(db, min_pending=5))
            db.close()


if __name__ == '__main__':
    unittest.main()
