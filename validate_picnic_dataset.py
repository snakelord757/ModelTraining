import json,sqlite3,hashlib,statistics
from pathlib import Path
from transformers import AutoTokenizer
ROOT=Path(__file__).resolve().parent
data=ROOT/'data/picnic-pdf-reference-v1'
tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'models/Qwen3-1.7B'),local_files_only=True)
report={}
for family in ['sft','grounded']:
    report[family]={};seen={}
    for split in ['train','validation','test']:
        sizes=[];facts=set();questions=set()
        for line in (data/family/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            text=tokenizer.apply_chat_template(row['messages'],tokenize=False,add_generation_prompt=False,enable_thinking=False)
            ids=tokenizer(text,add_special_tokens=False)['input_ids']
            assert tokenizer.eos_token_id in ids
            assert ids and '\ufffd' not in text
            sizes.append(len(ids));facts.add(row['fact_id'])
            questions.add(hashlib.sha256(row['messages'][1]['content'].encode()).hexdigest())
        seen[split]=(facts,questions)
        report[family][split]={'rows':len(sizes),'max_tokens':max(sizes),'median_tokens':statistics.median(sizes),'over_12288':sum(x>12288 for x in sizes)}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not seen[a][0]&seen[b][0], 'Fact leakage'
        assert not seen[a][1]&seen[b][1], 'Prompt leakage'
db=sqlite3.connect(data/'retrieval.sqlite')
report['retrieval_tests']=[]
for query,expected in [('Герберт',47),('комариные AND плеши',59),('Арчибальд*',68)]:
    rows=db.execute('SELECT id,page FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT 30',(query,)).fetchall()
    assert any(int(r[1])==expected for r in rows),(query,rows)
    report['retrieval_tests'].append({'query':query,'expected_page':expected,'found':True})
db.close()
report['status']='passed'
(data/'validation-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
