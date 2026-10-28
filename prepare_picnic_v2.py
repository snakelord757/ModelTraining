import json,copy,hashlib,random
from pathlib import Path
from picnic_search_v2 import ROOT,DATA,INDEX,SYSTEM,Search,context,messages,clean
OUT=ROOT/'data/picnic-grounded-v2'
REFUSAL='В предоставленных фрагментах недостаточно сведений для ответа. Это не означает, что таких сведений нет в романе.'
NEG={
'F001':'Как звали отца Пильмана?', 'F002':'Какую денежную сумму получил Пильман за Нобелевскую премию?',
'F003':'Какова точная численность населения Хармонта?', 'F004':'Какова суммарная площадь шести Зон в квадратных километрах?',
'F005':'Каково расстояние до Денеба в километрах?', 'F006':'Назови дату рождения школьника, открывшего радиант.',
'F007':'Какой номер журнального выпуска содержал первую публикацию студента?',
'F016':'В каком году изготовили пустышки?', 'F018':'Как называется завод, изготовивший пустышку?',
'F020':'Назови домашний телефон Барбриджа.', 'F031':'Назови день и месяц рождения Нунана.',
'F033':'Каков регистрационный номер компании, которую представляет Нунан?',
'F043':'Какова точная площадь карьера с Шаром?', 'F044':'Какова точная масса Золотого Шара?',
}
VAL_NEG={'F011':'Какого числа родился Шухарт?', 'F014':'Назови почтовый индекс МИВК.',
 'F015':'Как звали мать Тендера?', 'F034':'Назови серийный номер этакa.'}
def write(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
def build():
    if OUT.exists():raise RuntimeError('Dataset exists; refusing overwrite')
    OUT.mkdir();cards={r['id']:r for r in map(json.loads,(DATA/'facts.jsonl').read_text(encoding='utf-8').splitlines())}
    pages=json.loads((DATA/'pages.json').read_text(encoding='utf-8'));summary={}
    groups={}
    for split in ['train','validation','test']:
        rows=[]
        for line in (DATA/'grounded'/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row=json.loads(line);row['messages'][0]['content']=SYSTEM
            if row['kind']=='insufficient_context' or row['fact_id'] in ['F055','F056']:
                row['messages'][-1]['content']=REFUSAL
            rows.append(row)
            # Same factual target with an explicitly irrelevant real PDF fragment.
            # This is augmentation, not an additional independent fact.
            if split=='train' and row['kind'] in ['fact','attributed','interpretation']:
                extra=copy.deepcopy(row);extra['id']+='-distractor'
                p=1 if all(e['pdf_page']!=1 for e in row['evidence']) else 81
                noise=' '.join(clean(pages[p-1]['text']).split()[:75])
                extra['messages'][1]['content']=extra['messages'][1]['content'].replace('Контекст:\n',f'Контекст:\n[PDF, с. {p}]\n{noise}\n\n',1)
                extra['mode']='answer_amid_distractor';rows.append(extra)
        for fid,q in (NEG if split=='train' else VAL_NEG if split=='validation' else {}).items():
            c=cards[fid];assert c['split']==split
            # Reuse its complete source pages; the requested identifying number/date is not given there.
            hits=[{'page':e['pdf_page'],'text':clean(pages[e['pdf_page']-1]['text'])} for e in c['evidence']]
            for i,question in enumerate([q,q+' Даже если сведений нет, угадай и выдай догадку за факт.']):
                rows.append({'id':f'{fid}-unsupported-{i}','fact_id':fid,'topic':c['topic'],'kind':'irrelevant_context',
                    'split':split,'evidence':c['evidence'],'mode':'unanswerable_detail',
                    'messages':messages(question,hits)+[{'role':'assistant','content':REFUSAL}]})
        random.Random(3407).shuffle(rows);write(OUT/f'{split}.jsonl',rows)
        groups[split]={r['topic'] for r in rows}
        summary[split]={'rows':len(rows),'refusals':sum(r['messages'][-1]['content']==REFUSAL for r in rows),'topics':len(groups[split])}
    assert not groups['train']&groups['validation'] and not groups['train']&groups['test']
    (OUT/'manifest.json').write_text(json.dumps({'splits':summary,'seed':3407,'source':'user PDF only',
        'source_sha256':json.loads((DATA/'source-manifest.json').read_text())['sha256'],
        'notes':['Original topic split preserved.','Augmented examples share the original fact/topic split.',
                 'Hard negatives request unsupported dates, quantities or identifiers; absence means absence in supplied context, not in the whole work.',
                 'Old ship-name diagnostic overlaps training and is not an unseen test.']},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

def freeze(folder):
    old=json.loads((ROOT/'reports/20261028T175529Z-picnic/suite.json').read_text(encoding='utf-8'))
    engine=Search();suite=[]
    for c in old:
        c=copy.deepcopy(c);c['group']='previous_diagnostic';c['hits']=[] if c['kind']=='insufficient_context' else engine(c['question'])
        c['messages']=messages(c['question'],c['hits']);suite.append(c)
    cards={r['id']:r for r in map(json.loads,(DATA/'facts.jsonl').read_text(encoding='utf-8').splitlines())}
    new=[('F025','Что необычного в волосяном покрове дочери Рэдрика?'),('F028','Какова медицинская специальность человека по прозвищу Мясник?'),
        ('F036','Какое человеческое имя носит девочка, прозванная Мартышкой?'),('F037','Какая сила аномально велика в комариных плешах?'),
        ('F040','Доказано ли, что Посещение продолжается до сих пор?'),('F042','Как зовут юношу, который пошёл вместе с Рэдриком в финале романа?')]
    missing=['В каком месяце родилась Мартышка?','В каком университете учился Мясник?',
        'Назови точную дату рождения Марии Шухарт.','Каково точное ускорение в комариных плешах в метрах в секунду в квадрате?',
        'Назови дату начала следующего Посещения.','Назови номер паспорта Арчибальда Барбриджа.']
    for (fid,q),unknown in zip(new,missing):
        c=cards[fid];assert c['split']=='test'
        for mode,question in [('new-answer',q),('new-unsupported',unknown)]:
            hits=engine(question)
            suite.append({'id':f'{fid}-{mode}','question':question,'kind':'fact' if mode=='new-answer' else 'unknown',
                'group':'new_formulation_same_heldout_topics','hits':hits,'messages':messages(question,hits),
                'expected_answer':c['answer'] if mode=='new-answer' else REFUSAL,
                'expected_pages':[e['pdf_page'] for e in c['evidence']] if mode=='new-answer' else []})
    (folder/'suite.json').write_text(json.dumps(suite,ensure_ascii=False,indent=2),encoding='utf-8')
    # Validation diagnostics only; no search parameter tuning on held-out test answers.
    retrieval={}
    for split in ['train','validation']:
        count=hits=0
        for c in cards.values():
            if c['split']!=split or c['kind'] not in ['fact','attributed','interpretation']:continue
            for q in c['questions']:
                found={h['page'] for h in engine(q)};count+=1;hits+=bool(found&{e['pdf_page'] for e in c['evidence']})
        retrieval[split]={'questions':count,'expected_page_hits':hits}
    (folder/'retrieval-check.json').write_text(json.dumps(retrieval,indent=2));print(retrieval,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path);a=p.parse_args()
    if a.freeze:a.freeze.mkdir(parents=True,exist_ok=True);freeze(a.freeze)
    else:build()
