import json
import tempfile
import unittest
from pathlib import Path
from evaluate_literature import checks, compare, render_report, read_json, write_json, SUITE, SETTINGS
import hashlib

class EvaluationTests(unittest.TestCase):
    def test_detect_regression_and_improvement(self):
        case = {'id': 'x', 'exact': '391'}
        good = {'answer': '391', 'tokens': 1, 'hit_token_limit': False}
        bad = {'answer': 'Ответ: 392', 'tokens': 4, 'hit_token_limit': False}
        self.assertTrue(compare(case, good, bad)['lost'])
        self.assertTrue(compare(case, bad, good)['gained'])
        self.assertTrue(compare(case, good, good)['identical'])
        self.assertFalse(compare(case, bad, bad)['lost'])

    def test_format_checks(self):
        self.assertFalse(checks({'json_value': {'age': 30}}, '```json\n{"age":30}\n```')['Корректный JSON и значения'])
        self.assertTrue(checks({'json_value': {'age': 30}}, '{"age":30}')['Корректный JSON и значения'])
        self.assertFalse(checks({'paragraphs': 3}, 'one\n\ntwo')['Число абзацев'])

    def test_report_completeness_timing_and_escaping(self):
        cases = read_json(SUITE)
        self.assertEqual(len(cases), 20)
        doc = {'suite_sha256': hashlib.sha256(SUITE.read_bytes()).hexdigest(),
               'settings': SETTINGS, 'generation_seconds': 20,
               'answers': [{'id': c['id'], 'answer': '<script>alert(1)</script>',
                            'tokens': 5, 'hit_token_limit': False} for c in cases]}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / 'prompts.json').write_bytes(SUITE.read_bytes())
            write_json(folder / 'base.json', doc)
            write_json(folder / 'tuned.json', doc)
            render_report(folder, {'training_seconds': 3661, 'total_seconds': 3700}, 3800)
            report = (folder / 'report.html').read_text(encoding='utf-8')
            self.assertIn('01:01:01', report)
            self.assertNotIn('<script>', report)
            self.assertEqual(len(read_json(folder / 'comparison.json')['comparisons']), 20)
            doc['answers'].pop()
            write_json(folder / 'tuned.json', doc)
            with self.assertRaises(ValueError):
                render_report(folder, {}, 3800)

if __name__ == '__main__':
    unittest.main()
