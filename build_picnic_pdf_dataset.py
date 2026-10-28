"""Build a small, source-grounded reference dataset from the user's PDF.
The draft supplies candidate QAs only; evidence and all output citations are PDF-only.
"""
import json,re,hashlib,random,collections,html,sqlite3
from pathlib import Path
from picnic_facts import FACTS
ROOT=Path(__file__).resolve().parent/'data/picnic-pdf-reference-v1'
pages=json.loads((ROOT/'pages.json').read_text(encoding='utf-8'))
manifest=json.loads((ROOT/'source-manifest.json').read_text(encoding='utf-8'))
aligned=json.loads((ROOT/'alignment-review.json').read_text(encoding='utf-8'))
def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')
def clean(text):
    text=re.sub(r'(?<=\w)-\s*\n\s*(?=\w)','',text)
    text=re.sub(r'Библиотека сайта «Вселенная Братьев Стругацких»\s*-\s*strugatskie.com','',text)
    return re.sub(r'\s+',' ',text).strip()
def chapter(page):
    return 'Пролог' if page<3 else 'Глава 1' if page<24 else 'Глава 2' if page<45 else 'Глава 3' if page<68 else 'Глава 4'
# Reviewed page corrections where this edition differs from the online draft.
overrides={15:[4],16:[3,4],17:[3],18:[3],19:[13],20:[41,49],26:[31],27:[31],29:[30],32:[47],35:[59],37:[59],41:[68],43:[81],44:[81],46:[77,82]}
corrections={15:'Тендер — лаборант Кирилла.',7:'По словам Пильмана, координаты впервые опубликовал студент.'}
cards=[]
for i,(topic,kind,sid,nums,questions,answer) in enumerate(FACTS,1):
    if i==24:continue # Name list in the earlier edition is not used without matching PDF evidence.
    questions=[q.replace('Артуром','Арчибальдом').replace('Артуру','Арчибальду').replace('Артур','Арчибальд') for q in questions]
    answer=corrections.get(i,answer).replace('Артуром','Арчибальдом').replace('Артур','Арчибальд')
    evidence_pages=overrides.get(i,aligned[i-1]['pdf_pages'])
    if kind!='out_of_scope' and not evidence_pages:raise ValueError(f'No PDF evidence for {i}')
    cards.append({'id':f'F{i:03d}','topic':topic,'kind':kind,'questions':questions,'answer':answer,
        'evidence':[{'source_id':'user-pdf','pdf_page':p,'section':chapter(p)} for p in evidence_pages],
        'verification':'checked_against_user_pdf' if kind!='out_of_scope' else 'scope_policy_not_book_fact',
        'evidence_scope':'Absence claim: whole-work scope; cited page is background, not proof of absence.' if i in [55,56] else 'direct_or_explicitly_attributed'} )
# Group by topic: all paraphrases and closely related facts stay in the same split.
topics=sorted({r['topic'] for r in cards});random.Random(3407).shuffle(topics)
n=len(topics);train_topics=set(topics[:int(n*.7)]);val_topics=set(topics[int(n*.7):int(n*.85)])
for row in cards: row['split']='train' if row['topic'] in train_topics else 'validation' if row['topic'] in val_topics else 'test'
write_jsonl(ROOT/'facts.jsonl',cards)
system='Ты справочник по роману «Пикник на обочине» в предоставленном PDF. Отвечай кратко по существу. Не используй фильмы, игры и фанфики. Отделяй факты от слов персонажей, гипотез и интерпретаций. Не выдумывай неизвестные сведения. Ссылайся на страницы PDF. Возможны спойлеры.'
grounded_system=system+' Используй только приложенный контекст. Если в нём недостаточно сведений, прямо сообщи об этом; не подменяй отсутствие контекста утверждением, что факта нет в книге.'
closed=collections.defaultdict(list);rag=collections.defaultdict(list)
for card in cards:
    nums=[e['pdf_page'] for e in card['evidence']]
    cited=card['answer']+(' [PDF, с. '+', '.join(map(str,nums))+']' if nums else '')
    context='\n\n'.join(f'[PDF, с. {p}]\n{clean(pages[p-1]["text"])}' for p in nums)
    for v,question in enumerate(card['questions'],1):
        meta={'id':f'{card["id"]}-q{v}','fact_id':card['id'],'topic':card['topic'],'kind':card['kind'],'evidence':card['evidence'],'split':card['split']}
        closed[card['split']].append({**meta,'messages':[{'role':'system','content':system},{'role':'user','content':question},{'role':'assistant','content':cited}]})
        rag[card['split']].append({**meta,'id':meta['id']+'-grounded','mode':'context_answer',
            'messages':[{'role':'system','content':grounded_system},{'role':'user','content':f'Контекст:\n{context or "Вопрос вне области справочника."}\n\nВопрос: {question}'},{'role':'assistant','content':('В предоставленном фрагменте нет этих сведений. Устанавливать их по догадке нельзя.' if card['id'] in ('F055','F056') else cited)}]})
    if card['kind'] not in ('out_of_scope','unknown'):
        rag[card['split']].append({'id':card['id']+'-no-context','fact_id':card['id'],'topic':card['topic'],'kind':'insufficient_context','split':card['split'],'evidence':[],
            'messages':[{'role':'system','content':grounded_system},{'role':'user','content':'Контекст: не найден.\nВопрос: '+card['questions'][0]},
                        {'role':'assistant','content':'В предоставленном контексте нет сведений для ответа. Нужен соответствующий фрагмент романа.'}]})
for name,data in [('sft',closed),('grounded',rag)]:
    d=ROOT/name;d.mkdir(exist_ok=True)
    for split in ('train','validation','test'):write_jsonl(d/f'{split}.jsonl',data[split])
# Retrieval corpus covers the entire supplied book, not just the seed facts.
chunks=[]
for page in pages:
    words=clean(page['text']).split()
    for j,start in enumerate(range(0,len(words),280),1):
        part=words[start:start+340]
        if not part:continue
        chunks.append({'id':f'{page["id"]}-c{j:02d}','source_id':'user-pdf','pdf_page':page['page'],
                       'section':chapter(page['page']),'text':' '.join(part)})
write_jsonl(ROOT/'retrieval_chunks.jsonl',chunks)
db=sqlite3.connect(ROOT/'retrieval.sqlite')
db.execute('DROP TABLE IF EXISTS chunks')
db.execute("CREATE VIRTUAL TABLE chunks USING fts5(id UNINDEXED, page UNINDEXED, section UNINDEXED, text, tokenize='unicode61')")
db.executemany('INSERT INTO chunks VALUES (?,?,?,?)',[(r['id'],r['pdf_page'],r['section'],r['text']) for r in chunks]);db.commit();db.close()
checks=[]
for family in [closed,rag]:
    seen=set()
    for split,rows in family.items():
        for r in rows:
            assert r['id'] not in seen;seen.add(r['id'])
            assert [m['role'] for m in r['messages']]==['system','user','assistant']
            assert all(m['content'].strip() for m in r['messages'])
            assert all(1<=e['pdf_page']<=len(pages) for e in r['evidence'])
    sets=[{r['topic'] for r in family[s]} for s in ['train','validation','test']]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    checks.append('unique ids, nonempty messages, valid PDF pages, disjoint topics')
summary={'source_sha256':manifest['sha256'],'source_pages':len(pages),'facts':len(cards),
 'sft':{k:len(v) for k,v in closed.items()},'grounded':{k:len(v) for k,v in rag.items()},
 'retrieval_chunks':len(chunks),'status':'seed_dataset_not_exhaustive','checks':checks,
 'limitations':['Edition-specific names: Archibald, not Arthur.','Page 3,24,45,68 contain section boundaries; section labels use page starts and may include preceding text.',
 'PDF text extraction can contain spacing/recognition artifacts; original PDF is authoritative.',
 'Closed-book held-out topics test unseen knowledge; use grounded evaluation to test reading and refusal.',
 'Retrieval corpus intentionally contains held-out facts: retrieval access is not SFT training leakage.',
 'Automatic checks validate structure, not every semantic claim. This is a curated starting set, not a complete fan wiki.']}
(ROOT/'dataset-report.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
rows=['<!doctype html><html lang="ru"><meta charset="utf-8"><title>Пикник: датасет</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto}td,th{padding:10px;border:1px solid #ddd;vertical-align:top}table{border-collapse:collapse}pre{white-space:pre-wrap}</style>',
 '<h1>Справочник по предоставленному PDF</h1><p>Только роман. Спойлеры. Нумерация — страницы PDF от 1 до 83. Персонаж в этой версии: Арчибальд Барбридж.</p>',
 '<p>Начальный проверенный набор, не полная энциклопедия. Для справочника рекомендуются контекстные примеры grounded вместе с поиском по retrieval_chunks.jsonl. Старые источники из интернета не используются в выходных ссылках.</p>',
 '<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><table><tr><th>ID / часть</th><th>Вопрос</th><th>Ответ</th><th>Тип / страницы</th></tr>']
for r in cards:rows.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['id']+' / '+r['split'],r['questions'][0],r['answer'],r['kind']+' / '+','.join(str(e['pdf_page']) for e in r['evidence'])])+'</tr>')
rows.append('</table></html>');(ROOT/'dataset-preview.html').write_text('\n'.join(rows),encoding='utf-8')
(ROOT/'README.txt').write_text('''Источник: только PDF пользователя, см. source-manifest.json. Номера страниц начинаются с 1.
facts.jsonl: краткие справочные карточки с типом утверждения и PDF-страницами.
sft/: вопрос-ответ без контекста. Подходит для небольшого эксперимента, не гарантирует запоминание книги.
grounded/: рекомендуемый формат обучения справочника по найденному контексту + отказы при его отсутствии.
Не смешивайте train с validation/test. Перефразировки и связанные факты одной темы находятся в одном разделе.
retrieval_chunks.jsonl и retrieval.sqlite: полный корпус для поиска, не обучающие ответы. Не обучайте модель просто продолжать эти фрагменты.
PDF остаётся авторитетным источником: извлечение текста может содержать пробелы внутри слов и опечатки распознавания.
Набор небольшой и не охватывает все факты романа. История персонажей из фильмов, игр и фанфиков исключена.
В этой версии используется имя Арчибальд. Нельзя автоматически заменять его именем из другого издания.
Оценка: правильность факта, поддержка контекстом, верная страница, отсутствие выдумок, корректный отказ, отсутствие повторов.
Сначала проверить исходную Qwen с поиском. Затем сравнить небольшой адаптер с той же базой и одинаковым поиском.
Текущее литературное обучение не изменено. Новое обучение на книге не запускалось.
''',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
