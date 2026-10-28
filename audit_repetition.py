import json, re, collections, hashlib, statistics
from pathlib import Path
root = Path(__file__).resolve().parent
folder = root / 'reports/20261028T102148588898Z'
rows = [json.loads(x) for x in (root/'data/russian_classic_literature/train_dataset.json').read_text(encoding='utf-8').splitlines() if x.strip()]
prompts = collections.Counter()
hashes = collections.Counter()
repetition = []
examples = []
for row in rows:
    messages = row['messages']
    prompt = '\n'.join(m['content'] for m in messages if m['role'] != 'assistant')
    answer = messages[-1]['content']
    prompts[prompt] += 1
    hashes[hashlib.sha256(answer.encode()).hexdigest()] += 1
    words = re.findall(r'\w+', answer.casefold())
    ngrams = collections.Counter(tuple(words[i:i+10]) for i in range(max(0,len(words)-9)))
    repetition.append(1-len(ngrams)/sum(ngrams.values()) if ngrams else 0)
    if len(examples)<3:
        examples.append({'prompt':prompt, 'answer_start':answer[:400], 'answer_end':answer[-150:]})
result = {'rows':len(rows),'unique_prompts':len(prompts),'top_prompts':prompts.most_common(5),
          'duplicate_answers':sum(n-1 for n in hashes.values()),
          'source_repeat_10gram_median':statistics.median(repetition),
          'source_repeat_10gram_over_half':sum(r>.5 for r in repetition),'examples':examples}
for name in ['base','tuned']:
    answers=json.loads((folder/f'{name}.json').read_text(encoding='utf-8'))['answers']
    stats=[]
    for a in answers:
        words=re.findall(r'\w+',a['answer'].casefold())
        grams=[tuple(words[i:i+10]) for i in range(max(0,len(words)-9))]
        stats.append({'id':a['id'],'repeat_10gram':round(1-len(set(grams))/len(grams),3) if grams else 0,'limit':a['hit_token_limit']})
    result[name]=stats
state=json.loads((root/'outputs/qwen3-1.7b-literature-12k/checkpoint-1003/trainer_state.json').read_text())
logs=[x for x in state['log_history'] if 'loss' in x]
result['training']={'first50_mean_loss':statistics.mean(x['loss'] for x in logs[:50]),
                    'last50_mean_loss':statistics.mean(x['loss'] for x in logs[-50:]),
                    'max_grad_norm':max(x['grad_norm'] for x in logs),
                    'evaluation_entries':sum('eval_loss' in x for x in state['log_history'])}
from transformers import AutoTokenizer
from datasets import load_from_disk
tokenizer=AutoTokenizer.from_pretrained(str(root/'models/Qwen3-1.7B'),local_files_only=True)
data=load_from_disk(str(root/'data/prepared-qwen3-1.7b-whole-12288-0'))['train']
bad_eos=0
for row in data:
    if not any(t==tokenizer.eos_token_id and label==t for t,label in zip(row['input_ids'],row['labels'])):
        bad_eos+=1
first=data[0]
result['labels']={'missing_trainable_eos':bad_eos,
    'first_target_start':tokenizer.decode([t for t in first['labels'] if t!=-100][:35]),
    'first_target_end':tokenizer.decode([t for t in first['labels'] if t!=-100][-35:])}
(folder/'repetition-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
