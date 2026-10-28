"""Read-only adapter diagnosis: isolate decoding from weight changes."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from unsloth import FastModel
import torch, json, re, time
import transformers.modeling_utils as mu
from transformers import set_seed
from peft import PeftModel
from contextlib import nullcontext
from model_settings import ROOT, MODEL_PATH, OUTPUT

folder=ROOT/'reports/20261028T102148588898Z'
original=mu.safe_open
def pread(filename,*a,**kw):
    if str(filename).startswith(str(MODEL_PATH)): kw['backend']='pread'
    return original(filename,*a,**kw)
mu.safe_open=pread
model,processor=FastModel.from_pretrained(model_name=str(MODEL_PATH),max_seq_length=2048,
    dtype=torch.bfloat16,load_in_4bit=True,device_map={'':0},offload_embedding=False,
    text_only=True,local_files_only=True)
model=PeftModel.from_pretrained(model,str(ROOT/OUTPUT/'adapter'),is_trainable=False)
FastModel.for_inference(model)
model.eval()
tokenizer=getattr(processor,'tokenizer',processor)
cases=[c for c in json.loads((folder/'prompts.json').read_text()) if c['id'] in ('L02','C03')]
conditions=[('base_greedy',False,{'do_sample':False,'repetition_penalty':1.0}),
            ('tuned_greedy',True,{'do_sample':False,'repetition_penalty':1.0}),
            ('tuned_sampling',True,{'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.0}),
            ('tuned_sampling_penalty',True,{'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.1})]
results=[]
for name,adapter,settings in conditions:
    for case in cases:
        set_seed(3407)
        prompt=tokenizer.apply_chat_template([{'role':'user','content':case['prompt']}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        inputs=tokenizer(prompt,add_special_tokens=False,return_tensors='pt').to('cuda')
        started=time.monotonic()
        with (nullcontext() if adapter else model.disable_adapter()),torch.inference_mode():
            output=model.generate(**inputs,**settings,max_new_tokens=384,use_cache=True,pad_token_id=tokenizer.pad_token_id)
        tokens=output[0,inputs['input_ids'].shape[-1]:].tolist()
        answer=tokenizer.decode(tokens,skip_special_tokens=True).strip()
        words=re.findall(r'\w+',answer.casefold())
        grams=[tuple(words[i:i+10]) for i in range(max(0,len(words)-9))]
        row={'condition':name,'id':case['id'],'settings':settings,'seed':3407,'answer':answer,
             'tokens':len(tokens),'repeat_10gram':1-len(set(grams))/len(grams) if grams else 0,
             'seconds':time.monotonic()-started}
        results.append(row)
        (folder/'decoding-probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(row,ensure_ascii=False),flush=True)
