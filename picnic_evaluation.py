import json,re,time,html,hashlib,difflib
from pathlib import Path
from picnic_search import DATA,ROOT,search,context,system_prompt
def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
def prepare_suite(folder):
    cases=[]
    for line in (DATA/'grounded/test.jsonl').read_text(encoding='utf-8').splitlines():
        row=json.loads(line);question=row['messages'][1]['content'].rsplit('Вопрос:',1)[-1].strip()
        hits=[] if row['kind']=='insufficient_context' else search(question)
        cases.append({'id':row['id'],'question':question,'kind':row['kind'],'hits':hits,
            'expected_answer':row['messages'][-1]['content'],'expected_pages':[e['pdf_page'] for e in row['evidence']]})
    for i,q in enumerate(['Как называется корабль посетителей?','Сколько членов экипажа было на корабле посетителей?'],1):
        cases.append({'id':f'unknown-{i}','question':q,'kind':'unknown','hits':search(q),'expected_answer':'Книга не даёт подтверждённого ответа.','expected_pages':[]})
    assert len(cases)==20
    for case in cases:
        case['messages']=[{'role':'system','content':system_prompt()},
            {'role':'user','content':f'Контекст:\n{context(case["hits"])}\n\nВопрос: {case["question"]}'}]
    save(folder/'suite.json',cases)
    return cases
def metrics(answer,case):
    words=re.findall(r'\w+',answer.casefold());grams=[tuple(words[i:i+10]) for i in range(max(0,len(words)-9))]
    repeat=1-len(set(grams))/len(grams) if grams else 0
    cited=set()
    for match in re.finditer(r'\[PDF,\s*с\.\s*([\d,\s–-]+)\]',answer,re.I):
        cited.update(map(int,re.findall(r'\d+',match.group(1))))
    available={h['page'] for h in case['hits']}
    refusal=bool(re.search(r'недостаточ|нет (?:этих )?(?:сведений|данных|информации)|не (?:найден|указан|содержит)|не да[её]т',answer,re.I))
    return {'repeated_10gram_fraction':repeat,'cited_pages':sorted(cited),
        'citations_outside_context':sorted(cited-available),'refusal_marker':refusal,
        'expected_page_retrieved':bool(set(case['expected_pages'])&available) if case['expected_pages'] else None}
def evaluate(folder,phase,adapter=None):
    from picnic_inference import Model,SETTINGS
    model=Model(adapter);suite=load(folder/'suite.json')
    result={'phase':phase,'adapter':str(adapter) if adapter else None,'settings':SETTINGS,'seed':3407,
        'suite_sha256':hashlib.sha256((folder/'suite.json').read_bytes()).hexdigest(),'answers':[]}
    for i,case in enumerate(suite,1):
        answer=model.answer(case['messages'])
        result['answers'].append({'id':case['id'],**answer,'checks':metrics(answer['answer'],case)})
        save(folder/f'{phase}.json',result)
        print(f'{phase}: {i}/{len(suite)} | {answer["tokens"]} tokens | {answer["seconds"]:.1f}s',flush=True)
def report(folder,training,elapsed):
    suite=load(folder/'suite.json');base=load(folder/'base.json');tuned=load(folder/'tuned.json')
    assert base['settings']==tuned['settings'] and base['suite_sha256']==tuned['suite_sha256']
    assert [r['id'] for r in base['answers']]==[r['id'] for r in tuned['answers']]==[c['id'] for c in suite]
    from training_progress import format_duration
    esc=lambda x:html.escape(str(x))
    rows=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пикник: RAG до и после</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:30px auto;padding:18px}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{vertical-align:top;border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:16px 0}</style>',
        '<h1>Справочник по «Пикнику на обочине»: до и после</h1>',
        f'<p>Обучение: <b>{format_duration(training["training_seconds"])}</b>; весь цикл: {format_duration(elapsed)}. '
        f'Эпох: {training["epoch"]}; шагов: {training["global_step"]}; лучший checkpoint: {esc(training["best_model_checkpoint"])}.</p>',
        f'<p>Валидационный loss: {training["initial_validation"]["eval_loss"]:.4f} → {training["final_validation"]["eval_loss"]:.4f}. Снижение loss не доказывает улучшение справочных ответов.</p>',
        '<p>18 удержанных тестовых примеров (включая 6 проверок без контекста) и 2 новых вопроса без установленного ответа. Это 20 заданий, часть — перефразировки. Поиск и контекст одинаковы до/после; эталонные ответы и страницы не используются поиском.</p>',
        '<p>Автопроверки выявляют повторы, ссылки за пределами контекста и маркеры отказа. Совпадение страницы не доказывает поддержку утверждения. Фактическую точность и смысл нужно проверять по тексту. Наличие нужной страницы оценивает поиск отдельно от генерации.</p>']
    regressions=[];comparisons=[]
    for case,a,b in zip(suite,base['answers'],tuned['answers']):
        reasons=[]
        if b['hit_limit'] and not a['hit_limit']:reasons.append('ответ впервые достиг лимита генерации')
        if b['checks']['repeated_10gram_fraction']>=.2 and a['checks']['repeated_10gram_fraction']<.2:reasons.append('появилось зацикливание')
        if b['checks']['citations_outside_context'] and not a['checks']['citations_outside_context']:reasons.append('появились ссылки вне контекста')
        if case['kind']=='insufficient_context' and a['checks']['refusal_marker'] and not b['checks']['refusal_marker']:reasons.append('исчез маркер отказа при отсутствии контекста')
        if reasons:regressions.append({'id':case['id'],'reasons':reasons})
        same=a['answer']==b['answer'];comparisons.append({'id':case['id'],'identical':same,'possible_regressions':reasons})
        rows.extend([f'<h2>{esc(case["id"])}: {esc(case["question"])}</h2>',
            f'<p>Тип: {esc(case["kind"])}. Найденные страницы: {esc([h["page"] for h in case["hits"]])}. Эталонные: {esc(case["expected_pages"])}.</p>',
            f'<p>Эталон: {esc(case["expected_answer"])}</p>',
            f'<p>Ответ {"совпадает" if same else "изменился"}. Возможные ухудшения: {esc(reasons or "автоматически не выявлены; нужна смысловая проверка")}.</p>',
            '<table><tr><th>Базовая Qwen + поиск</th><th>Адаптер + тот же поиск</th></tr>',
            f'<tr><td><pre>{esc(a["answer"])}</pre></td><td><pre>{esc(b["answer"])}</pre></td></tr></table>',
            f'<p>Проверки до: {esc(a["checks"])}; после: {esc(b["checks"])}. Достигнут лимит: {a["hit_limit"]} / {b["hit_limit"]}.</p>',
            '<details><summary>Изменения текста</summary><pre>'+esc('\n'.join(difflib.unified_diff(a['answer'].splitlines(),b['answer'].splitlines(),lineterm='')))+'</pre></details>',
            '<details><summary>Одинаковый найденный контекст</summary><pre>'+esc(context(case['hits']))+'</pre></details>'])
    rows.insert(7,'<p>Возможные ухудшения по автоматическим проверкам: '+esc([r['id'] for r in regressions] or 'не выявлены')+'</p>')
    review_path=folder/'semantic-review.json'
    if review_path.exists():
        review=load(review_path)
        block='<section><h2>Содержательная проверка ответов</h2><p>'+esc(review['summary'])+'</p>'
        block+='<p>'+esc(review['limitations'])+'</p><ul>'
        block+=''.join('<li><b>'+esc(r['id'])+'</b> — '+esc(r['assessment'])+'</li>' for r in review['cases'])
        rows.insert(7,block+'</ul></section>')
    (folder/'report.html').write_text('\n'.join(rows)+'</html>',encoding='utf-8')
    save(folder/'comparison.json',{'training_seconds':training['training_seconds'],'pipeline_seconds':elapsed,'regressions':regressions,'comparisons':comparisons})
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--phase',choices=['base','previous','tuned'],required=True);p.add_argument('--adapter',type=Path)
    args=p.parse_args();evaluate(args.folder,args.phase,args.adapter)
