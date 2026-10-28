import json,re,bisect
from pathlib import Path
from picnic_facts import FACTS
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/picnic-pdf-reference-v1'
norm=lambda t: re.sub(r'[^а-яa-z0-9]','',t.casefold().replace('ё','е'))
pages=json.loads((OUT/'pages.json').read_text(encoding='utf-8'))
offsets=[];joined=''
for p in pages:
    offsets.append(len(joined));joined+=norm(p['text'])
old={s:json.loads((ROOT/f'data/picnic-reference-v1/sources/{s}.json').read_text(encoding='utf-8'))['paragraphs'] for s in ['prologue','chapter-1','chapter-2','chapter-3','chapter-4']}
starts={'prologue':0}
for s in ['chapter-1','chapter-2','chapter-3','chapter-4']:
    anchor=norm(old[s][32]['text'])
    pos=joined.find(anchor)
    if pos<0: raise ValueError((s,anchor))
    starts[s]=pos
keys=list(starts)
ends={k:starts[keys[i+1]] if i+1<len(keys) else len(joined) for i,k in enumerate(keys)}
mapping=[]
for i,(topic,kind,sid,nums,qs,answer) in enumerate(FACTS,1):
    hits=[];unmatched=[]
    if sid:
        for n in nums:
            paragraph=old[sid][n-1]['text'];anchor=norm(paragraph)
            if len(anchor)<45:
                # Use a longer preceding context to disambiguate short dialogue lines.
                anchor=norm(' '.join(p['text'] for p in old[sid][max(32,n-3):n]))
            found=-1
            for length in [180,90,45]:
                if len(anchor)<length:continue
                found=joined.find(anchor[:length],starts[sid],ends[sid])
                if found>=0:break
            if found<0:unmatched.append(n);continue
            page=bisect.bisect_right(offsets,found)
            hits.append({'page':page,'anchor':anchor[:length],'old_paragraph':n})
    mapping.append({'id':f'F{i:03d}','topic':topic,'kind':kind,'question':qs[0], 'answer':answer,
                    'pdf_pages':sorted({h['page'] for h in hits}),'alignment':hits,'unmatched':unmatched})
(OUT/'alignment-review.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding='utf-8')
print('chapter pages', {k:bisect.bisect_right(offsets,v) for k,v in starts.items()})
for r in mapping: print(r['id'],r['pdf_pages'],r['unmatched'],r['question'],r['answer'])
