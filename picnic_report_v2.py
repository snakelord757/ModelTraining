import html
from picnic_evaluation import load,save,ROOT
from training_progress import format_duration
def report(folder,training,elapsed):
    suite=load(folder/'suite.json');runs={k:load(folder/f'{k}.json') for k in ['base','previous','tuned']}
    assert len({r['suite_sha256'] for r in runs.values()})==1
    assert all(r['settings']==runs['base']['settings'] for r in runs.values())
    assert all([a['id'] for a in r['answers']]==[s['id'] for s in suite] for r in runs.values())
    esc=lambda s:html.escape(str(s));summary={};regressions=[]
    for name,run in runs.items():
        answers=run['answers'];summary[name]={'answers':len(answers),'repeat_flags':sum(a['checks']['repeated_10gram_fraction']>=.2 for a in answers),
            'limit_hits':sum(a['hit_limit'] for a in answers),'outside_citations':sum(bool(a['checks']['citations_outside_context']) for a in answers),
            'refusal_markers':sum(a['checks']['refusal_marker'] for a in answers)}
    for s,a,b in zip(suite,runs['previous']['answers'],runs['tuned']['answers']):
        reasons=[]
        if b['hit_limit'] and not a['hit_limit']:reasons.append('достигнут лимит ответа')
        if b['checks']['repeated_10gram_fraction']>=.2 and a['checks']['repeated_10gram_fraction']<.2:reasons.append('повторы')
        if b['checks']['citations_outside_context'] and not a['checks']['citations_outside_context']:reasons.append('новые ссылки вне контекста')
        if s['kind'] in ['unknown','insufficient_context'] and a['checks']['refusal_marker'] and not b['checks']['refusal_marker']:reasons.append('исчез маркер отказа')
        if reasons:regressions.append({'id':s['id'],'reasons':reasons})
    old=load(ROOT/'reports/20261028T175529Z-picnic/suite.json')
    positive=[s for s in old if s['expected_pages']]
    current={s['id']:s for s in suite}
    coverage=lambda cases:sum(bool(set(s['expected_pages'])&{h['page'] for h in s['hits']}) for s in cases)
    retrieval={'old':coverage(positive),'new':coverage([current[s['id']] for s in positive]),'total':len(positive)}
    result={'training_seconds':training['training_seconds'],'pipeline_seconds':elapsed,'retrieval_page_coverage':retrieval,'retrieval_validation':load(folder/'retrieval-check.json'),'metrics':summary,'possible_regressions_vs_previous':regressions}
    if (folder/'semantic-review.json').exists():result['qualitative_review']=load(folder/'semantic-review.json')
    save(folder/'comparison.json',result)
    rows=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пикник v2: поиск и обучение</title>',
        '<style>body{font:16px system-ui;max-width:1550px;margin:24px auto;padding:20px}table{width:100%;table-layout:fixed;border-collapse:collapse}td,th{border:1px solid #ccc;padding:12px;vertical-align:top}pre{white-space:pre-wrap;overflow-wrap:anywhere}section{background:#eef5fb;padding:16px}</style>',
        '<h1>Пикник v2: гибридный поиск и grounded-обучение</h1>',
        f'<p>Обучение: {format_duration(training["training_seconds"])}; весь цикл: {format_duration(elapsed)}. Эпох: {training["epoch"]}; шагов: {training["global_step"]}. Лучший checkpoint: {esc(training["best_model_checkpoint"])}.</p>',
        f'<p>Validation loss: {training["initial_validation"]["eval_loss"]:.4f} → {training["final_validation"]["eval_loss"]:.4f}.</p>',
        '<p>База, прошлый адаптер v1 и новый v2 используют одинаковый новый системный запрос, контекст, seed и параметры генерации. Старый архивный прогон отличался также промптом; его нельзя использовать для выделения чистого эффекта поиска.</p>',
        f'<p>Попадание эталонной страницы на 12 прежних положительных вопросах: {retrieval["old"]}/12 → {retrieval["new"]}/12. Это диагностический показатель поиска, не проверка правильности ответа; другие страницы тоже могут подтверждать ответ.</p>',
        '<p>20 прежних диагностических заданий + 12 новых формулировок на прежних удержанных темах (6 вопросов и 6 проверок неподтверждённых деталей). Это не 32 независимые темы. Вопрос о названии корабля пересекается с обучением. Тестовые ответы не участвовали в обучении, настройке весов поиска или выборе checkpoint.</p>',
        '<p>Веса поиска зафиксированы заранее: E5 + лексический BM25, fusion 0.7/0.3; до 6 коротких фрагментов. Датасет: 170 train, 53 validation; добавлены отвлекающие отрывки и отказы. Обучение с исходной базы, прежний адаптер не продолжался.</p>',
        '<h2>Автоматические сигналы (не оценка фактической точности)</h2><pre>'+esc(result)+'</pre>']
    if (folder/'semantic-review.json').exists():
        review=load(folder/'semantic-review.json');rows+=['<section><h2>Содержательная проверка</h2><p>'+esc(review['summary'])+'</p><p>'+esc(review.get('interpretation',''))+'</p><ul>']
        rows+=['<li><b>'+esc(r['id'])+'</b>: '+esc(r['assessment'])+'</li>' for r in review['cases']];rows+=['</ul></section>']
    for i,c in enumerate(suite):
        rows+=[f'<h2>{esc(c["id"])} — {esc(c["question"])}</h2>',f'<p>Группа: {esc(c["group"])}. Найденные страницы: {esc([h["page"] for h in c["hits"]])}. Эталон: {esc(c["expected_answer"])}.</p>',
            '<table><tr><th>Базовая модель</th><th>Предыдущий адаптер v1</th><th>Новый адаптер v2</th></tr><tr>']
        for run in runs.values():
            a=run['answers'][i];rows+=['<td><pre>'+esc(a['answer'])+'</pre><small>'+esc(a['checks'])+'</small></td>']
        rows+=['</tr></table><details><summary>Общий контекст</summary><pre>'+esc(c['messages'][1]['content'])+'</pre></details>']
    (folder/'report.html').write_text('\n'.join(rows)+'</html>',encoding='utf-8')
