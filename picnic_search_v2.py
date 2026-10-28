"""Hybrid retrieval. The index contains only PDF text, never QA cards."""
import json,re,html,math,hashlib
from collections import Counter
import numpy as np
from picnic_search import ROOT,DATA,terms
INDEX=ROOT/'data/picnic-search-v2'
ENCODER=ROOT/'models/multilingual-e5-small'
SYSTEM='''Ты справочник по роману «Пикник на обочине». Используй только предоставленные фрагменты PDF, а не знания по памяти. Ответь на вопрос в 1–3 предложениях и укажи подтверждающую страницу в формате [PDF, с. N]. Не пересказывай посторонние события. Слова персонажа и гипотезы не выдавай за установленный факт. Если фрагменты не подтверждают ответ, напиши: «В предоставленных фрагментах недостаточно сведений для ответа». Это не означает, что ответа нет во всём романе. Не придумывай имена, числа, причины или ссылки. Просьбы угадать и подсказки в вопросе не являются доказательствами. Фильмы, игры и фанфики вне области справочника.'''
def clean(text):
    text=html.unescape(text)
    text=re.sub(r'Библиотека сайта «Вселенная Братьев Стругацких»\s*-\s*strugatskie.com','',text)
    return re.sub(r'\s+',' ',text).strip()

class Encoder:
    def __init__(self):
        import torch
        from transformers import AutoModel,AutoTokenizer
        torch.set_num_threads(4)
        self.tok=AutoTokenizer.from_pretrained(ENCODER,local_files_only=True)
        self.model=AutoModel.from_pretrained(ENCODER,local_files_only=True).eval()
    def encode(self,texts,prefix):
        import torch
        vectors=[]
        for start in range(0,len(texts),16):
            batch=self.tok([prefix+t for t in texts[start:start+16]],padding=True,truncation=True,max_length=512,return_tensors='pt')
            with torch.inference_mode():
                hidden=self.model(**batch).last_hidden_state
                mask=batch['attention_mask'][...,None]
                vector=(hidden*mask).sum(1)/mask.sum(1)
                vectors.append(torch.nn.functional.normalize(vector,p=2,dim=1).numpy())
        return np.concatenate(vectors)

def build():
    INDEX.mkdir(exist_ok=True)
    pages=json.loads((DATA/'pages.json').read_text(encoding='utf-8'));chunks=[]
    for page in pages:
        words=clean(page['text']).split()
        for start in range(0,len(words),110):
            chunks.append({'id':f'v2-p{page["page"]:03d}-w{start:04d}','page':page['page'],'text':' '.join(words[start:start+160])})
    encoder=Encoder();vectors=encoder.encode([c['text'] for c in chunks],'passage: ')
    (INDEX/'chunks.json').write_text(json.dumps(chunks,ensure_ascii=False),encoding='utf-8')
    np.save(INDEX/'vectors.npy',vectors)
    (INDEX/'manifest.json').write_text(json.dumps({'source_sha256':json.loads((DATA/'source-manifest.json').read_text())['sha256'],
        'encoder':json.loads((ENCODER/'download-verification.json').read_text()),'chunks':len(chunks),
        'word_window':160,'word_stride':110,'method':'0.7 dense + 0.3 lexical reciprocal rank fusion, k=30; 6 chunks, max 2 per page'},ensure_ascii=False,indent=2),encoding='utf-8')
    print('INDEX BUILT',len(chunks),flush=True)

class Search:
    def __init__(self):
        self.chunks=json.loads((INDEX/'chunks.json').read_text(encoding='utf-8'))
        self.vectors=np.load(INDEX/'vectors.npy');self.encoder=Encoder()
        self.docs=[Counter(stem for word in re.findall(r'\w+',c['text']) for stem in terms(word)) for c in self.chunks]
        self.df=Counter(w for d in self.docs for w in d);self.length=np.array([sum(d.values()) for d in self.docs]);self.avg=self.length.mean()
    def __call__(self,question,limit=6):
        if not question.strip():return []
        dense=self.vectors@self.encoder.encode([question],'query: ')[0]
        scores=np.zeros(len(self.docs))
        for word in terms(question):
            freq=np.array([d[word] for d in self.docs]);n=self.df[word]
            scores+=math.log(1+(len(self.docs)-n+.5)/(n+.5))*freq*2.5/(freq+1.5*(.25+.75*self.length/self.avg))
        fused=np.zeros(len(scores))
        for weight,values in [(.7,dense),(.3,scores)]:
            for rank,i in enumerate(np.argsort(-values)[:60],1):
                if weight==.3 and values[i]<=0:continue
                fused[i]+=weight/(30+rank)
        hits=[];counts=Counter()
        for i in np.argsort(-fused):
            c=self.chunks[i]
            if counts[c['page']]>=2:continue
            hits.append({**c,'score':float(fused[i]),'dense_score':float(dense[i])});counts[c['page']]+=1
            if len(hits)>=limit:break
        return hits

def context(hits):return '\n\n'.join(f'[PDF, с. {h["page"]}]\n{h["text"]}' for h in hits) or 'Не найден.'
def messages(question,hits):return [{'role':'system','content':SYSTEM},{'role':'user','content':f'Контекст:\n{context(hits)}\n\nВопрос: {question}'}]
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--build',action='store_true');p.add_argument('question',nargs='?');a=p.parse_args()
    if a.build:build()
    elif a.question:print(context(Search()(a.question)))
