"""Local lexical retrieval over the user's PDF. No answer cards used for ranking."""
import re,json,sqlite3
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/picnic-pdf-reference-v1'
STOP=set('а и но в во на по к ко о об от до из за со с у для ли что как какой какая какие какое кто когда где чем кем его ее их он она это эта этой этом такой такие мне книгу книге романе роман пикник обочине ответь только пожалуйста назови расскажи является сколько есть было были быть чтобы если или при еще же не'.split())
def terms(question):
    words=re.findall(r'[а-яa-z0-9]+',question.casefold().replace('ё','е'))
    result=[]
    for word in words:
        if word in STOP or len(word)<3:continue
        stem=re.sub(r'(иями|ами|ями|ого|ему|ому|ыми|ими|ия|ие|ии|ой|ый|ий|ая|ую|ые|ам|ям|ах|ях|ом|ем|ов|ев|а|я|ы|и|у|ю|е)$','',word) if len(word)>4 else word
        result.append(stem if len(stem)>=3 else word)
    return list(dict.fromkeys(result))
def search(question,limit=4):
    tokens=terms(question)
    if not tokens:return []
    query=' OR '.join('"'+w+'"*' for w in tokens)
    db=sqlite3.connect(f'file:{DATA / "retrieval.sqlite"}?mode=ro',uri=True)
    rows=db.execute('SELECT id,page,section,text,bm25(chunks) FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT 40',(query,)).fetchall();db.close()
    result=[];pages=set()
    for cid,page,section,text,score in rows:
        if int(page) in pages:continue
        pages.add(int(page));result.append({'id':cid,'page':int(page),'section':section,'text':text,'score':score})
        if len(result)==limit:break
    return result
def context(hits):
    return '\n\n'.join(f'[PDF, с. {h["page"]}]\n{h["text"]}' for h in hits) or 'Не найден.'
def system_prompt():
    row=json.loads((DATA/'grounded/train.jsonl').read_text(encoding='utf-8').splitlines()[0])
    return row['messages'][0]['content']
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('question');args=p.parse_args()
    print(context(search(args.question)))
