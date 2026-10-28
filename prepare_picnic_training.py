import json,hashlib
from datasets import Dataset,DatasetDict
from transformers import AutoTokenizer
from model_settings import ROOT,MODEL_PATH
from picnic_search import DATA
import argparse
p=argparse.ArgumentParser();p.add_argument('--source',type=str,default=str(DATA/'grounded'));p.add_argument('--output',default='data/picnic-grounded-tokenized-v1');args=p.parse_args()
out=ROOT/args.output
if out.exists():raise SystemExit('Prepared output exists; refusing overwrite')
tokenizer=AutoTokenizer.from_pretrained(str(MODEL_PATH),local_files_only=True)
result={};metadata={'source':'user PDF grounded dataset','max_length':4096,'splits':{}}
for split in ['train','validation','test']:
    path=ROOT/args.source/f'{split}.jsonl';rows=[];maxlen=0
    for line in path.read_text(encoding='utf-8').splitlines():
        item=json.loads(line);messages=item['messages']
        full=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        prompt=tokenizer.apply_chat_template(messages[:-1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        assert full.startswith(prompt),'Template prefix mismatch'
        encoded=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
        ids=encoded['input_ids'];assert len(ids)<=4096,'No silent truncation'
        labels=[t if start>=len(prompt) and end>start else -100 for t,(start,end) in zip(ids,encoded['offset_mapping'])]
        assert tokenizer.eos_token_id in labels and any(t!=-100 for t in labels[1:])
        rows.append({'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels});maxlen=max(maxlen,len(ids))
    result[split]=Dataset.from_list(rows)
    metadata['splits'][split]={'rows':len(rows),'max_tokens':maxlen,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'target_tokens':sum(sum(t!=-100 for t in r['labels']) for r in rows)}
DatasetDict(result).save_to_disk(str(out))
(out/'preparation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(metadata,ensure_ascii=False,indent=2))
