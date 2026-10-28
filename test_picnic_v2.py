import json,unittest
from collections import Counter
from picnic_search_v2 import ROOT,DATA,INDEX,Search

class V2Tests(unittest.TestCase):
    def test_dataset_integrity(self):
        groups={};ids=set()
        for split in ['train','validation','test']:
            rows=[json.loads(l) for l in (ROOT/f'data/picnic-grounded-v2/{split}.jsonl').read_text(encoding='utf-8').splitlines()]
            groups[split]={r['topic'] for r in rows}
            for r in rows:
                self.assertNotIn(r['id'],ids);ids.add(r['id'])
                self.assertEqual([m['role'] for m in r['messages']],['system','user','assistant'])
                self.assertTrue(all(1<=e['pdf_page']<=83 for e in r['evidence']))
        self.assertFalse(groups['train']&groups['validation']);self.assertFalse(groups['train']&groups['test'])
    def test_index_source_and_search(self):
        manifest=json.loads((INDEX/'manifest.json').read_text(encoding='utf-8'))
        source=json.loads((DATA/'source-manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['source_sha256'],source['sha256'])
        engine=Search();self.assertEqual(engine(''),[])
        hits=engine('Какую награду получил доктор Пильман?')
        self.assertEqual(len(hits),6);self.assertLessEqual(max(Counter(h['page'] for h in hits).values()),2)
        self.assertEqual(len({h['id'] for h in hits}),len(hits))
    def test_answer_only_labels(self):
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3-1.7B',local_files_only=True)
        data=load_from_disk(str(ROOT/'data/picnic-grounded-tokenized-v2'))
        for split in data.values():
            for r in split:
                self.assertLessEqual(len(r['input_ids']),4096)
                self.assertEqual(r['labels'][0],-100)
                self.assertIn(tok.eos_token_id,r['labels'])
                self.assertTrue(all(y==-100 or x==y for x,y in zip(r['input_ids'],r['labels'])))

if __name__=='__main__':unittest.main()
