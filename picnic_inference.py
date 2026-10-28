import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
from model_settings import ROOT,MODEL_PATH
SETTINGS={'max_new_tokens':384,'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,'repetition_penalty':1.0}
class Model:
    def __init__(self,adapter=None):
        from unsloth import FastModel
        import torch
        import transformers.modeling_utils as mu
        original=mu.safe_open
        def pread(filename,*a,**kw):
            if str(filename).startswith(str(MODEL_PATH)):kw['backend']='pread'
            return original(filename,*a,**kw)
        mu.safe_open=pread
        self.model,processor=FastModel.from_pretrained(model_name=str(MODEL_PATH),max_seq_length=4096,
            dtype=torch.bfloat16,load_in_4bit=True,device_map={'':0},offload_embedding=False,text_only=True,local_files_only=True)
        if adapter:
            from peft import PeftModel
            self.model=PeftModel.from_pretrained(self.model,str(adapter),is_trainable=False)
        FastModel.for_inference(self.model);self.model.eval()
        self.tokenizer=getattr(processor,'tokenizer',processor)
    def answer(self,messages):
        import torch,time
        from transformers import set_seed
        set_seed(3407)
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.tokenizer(text,add_special_tokens=False,return_tensors='pt').to('cuda')
        if inputs['input_ids'].shape[-1]+SETTINGS['max_new_tokens']>4096:raise ValueError('Context too long; reduce retrieved chunks')
        started=time.monotonic()
        with torch.inference_mode():out=self.model.generate(**inputs,**SETTINGS,use_cache=True,pad_token_id=self.tokenizer.pad_token_id)
        ids=out[0,inputs['input_ids'].shape[-1]:].tolist()
        return {'answer':self.tokenizer.decode(ids,skip_special_tokens=True).strip(),'tokens':len(ids),
                'hit_limit':len(ids)>=SETTINGS['max_new_tokens'] and ids[-1]!=self.tokenizer.eos_token_id,'seconds':time.monotonic()-started}
if __name__=='__main__':
    import argparse,json
    from picnic_search import search,context,system_prompt
    p=argparse.ArgumentParser();p.add_argument('--base',action='store_true');args=p.parse_args()
    adapter=None
    if not args.base:adapter=json.loads((ROOT/'picnic-active.json').read_text())['adapter']
    model=Model(adapter)
    print('Справочник по PDF. Введите вопрос; /exit — выход.',flush=True)
    while True:
        try:q=input('\nВопрос: ').strip()
        except (EOFError,KeyboardInterrupt):break
        if q=='/exit':break
        if not q:continue
        hits=search(q)
        messages=[{'role':'system','content':system_prompt()},{'role':'user','content':f'Контекст:\n{context(hits)}\n\nВопрос: {q}'}]
        print(model.answer(messages)['answer'],flush=True)
        print('Найденные страницы PDF:',', '.join(str(h['page']) for h in hits) or 'нет',flush=True)
