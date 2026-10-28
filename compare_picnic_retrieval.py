import json
from pathlib import Path
from picnic_search import DATA,search
folder=Path('reports/20261028T181312Z-picnic-v2')
current=json.loads((folder/'retrieval-check.json').read_text())
cards=[json.loads(l) for l in (DATA/'facts.jsonl').read_text(encoding='utf-8').splitlines()]
for split in ['train','validation']:
    hits=0
    for c in cards:
        if c['split']!=split or c['kind'] not in ['fact','attributed','interpretation']:continue
        for q in c['questions']:
            hits+=bool({h['page'] for h in search(q)}&{e['pdf_page'] for e in c['evidence']})
    current[split]['old_expected_page_hits']=hits
(folder/'retrieval-check.json').write_text(json.dumps(current,indent=2));print(current)
