"""Reproducible heuristic selection; no claim of human editorial review."""
import json, re, random, hashlib, collections
from datasets import load_from_disk, DatasetDict
from model_settings import ROOT

destination=ROOT/'data/qwen-selected-500-v1'
if destination.exists():
    raise SystemExit('Selection already exists; refusing to overwrite')
source=ROOT/'data/russian_classic_literature'
cached=load_from_disk(str(ROOT/'data/prepared-qwen3-1.7b-whole-12288-0'))
rng=random.Random(3407)
manifest={'seed':3407,'selection':'Whole examples, 512..6000 tokens, sentence-ending punctuation, Cyrillic start, <=5% repeated word 10-grams; balanced round-robin over author prompts. Heuristics, not human quality labels.', 'splits':{}}
used=set()
chosen={}
for split,limit in [('train',500),('validation',32)]:
    rows=[json.loads(x) for x in (source/f'{split}_dataset.json').read_text(encoding='utf-8').splitlines() if x.strip()]
    assert len(rows)==len(cached[split])
    buckets=collections.defaultdict(list)
    for i,row in enumerate(rows):
        text=row['messages'][-1]['content'].strip()
        words=re.findall(r'\w+',text.casefold())
        grams=[tuple(words[j:j+10]) for j in range(max(0,len(words)-9))]
        repetition=1-len(set(grams))/len(grams) if grams else 0
        clean=text.strip(' \n\r\t«»"\'—–-()')
        digest=hashlib.sha256(' '.join(words).encode()).hexdigest()
        if not 512<=len(cached[split][i]['input_ids'])<=6000: continue
        if not clean or not re.match('[А-ЯЁ]',clean) or not re.search(r'[.!?…][»"\')]*$',text): continue
        if repetition>.05 or digest in used: continue
        prompt=next(m['content'] for m in reversed(row['messages']) if m['role']=='user')
        buckets[prompt].append((i,digest))
    for bucket in buckets.values(): rng.shuffle(bucket)
    keys=sorted(buckets); rng.shuffle(keys)
    selected=[]
    while len(selected)<limit and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(selected)<limit:
                i,digest=buckets[key].pop()
                if digest in used: continue
                used.add(digest); selected.append(i)
    if len(selected)!=limit: raise RuntimeError(f'Only {len(selected)} suitable rows in {split}')
    chosen[split]=cached[split].select(selected)
    manifest['splits'][split]={'rows':len(selected),'source_indices':selected,
        'target_tokens':sum(sum(t!=-100 for t in row['labels']) for row in chosen[split])}
DatasetDict(chosen).save_to_disk(str(destination))
(destination/'preparation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
