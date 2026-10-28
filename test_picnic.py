import tempfile,unittest
from pathlib import Path
from picnic_search import search,terms
from picnic_evaluation import prepare_suite,metrics

class PicnicTests(unittest.TestCase):
    def test_query_and_sources(self):
        self.assertEqual(terms('" OR * ()'), [])
        hits=search('Редрик Шухарт')
        self.assertTrue(hits)
        self.assertEqual(len(hits),len({h['page'] for h in hits}))
        self.assertTrue(all(1<=h['page']<=83 and h['text'] for h in hits))

    def test_frozen_suite_and_missing_context(self):
        with tempfile.TemporaryDirectory() as folder:
            cases=prepare_suite(Path(folder))
        self.assertEqual(len(cases),20)
        missing=[c for c in cases if c['kind']=='insufficient_context']
        self.assertEqual(len(missing),6)
        self.assertTrue(all(not c['hits'] for c in missing))
        for c in cases:
            self.assertEqual([m['role'] for m in c['messages']],['system','user'])

    def test_regression_signals(self):
        case={'hits':[{'page':3}],'expected_pages':[3]}
        checks=metrics('Недостаточно данных. [PDF, с. 8]',case)
        self.assertEqual(checks['citations_outside_context'],[8])
        self.assertTrue(checks['refusal_marker'])
        checks=metrics(' '.join(['один два три четыре пять шесть семь восемь девять десять']*8),case)
        self.assertGreater(checks['repeated_10gram_fraction'],.2)

if __name__=='__main__':unittest.main()
