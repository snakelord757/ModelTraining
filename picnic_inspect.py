"""Compact read-only inspection of a completed or ongoing comparison."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--phase',default='tuned');p.add_argument('--start',type=int,default=0);a=p.parse_args()
suite=json.loads((a.folder/'suite.json').read_text(encoding='utf-8'))
run=json.loads((a.folder/f'{a.phase}.json').read_text(encoding='utf-8'))
for s,r in zip(suite[a.start:],run['answers'][a.start:]):
    print(s['id']+' | '+s['question']+'\n'+r['answer']+'\n')
