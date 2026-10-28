"""Fetch public reading pages; preserve provenance, not a claim of an open license."""
import urllib.request, hashlib, json, re, datetime
from html.parser import HTMLParser
from pathlib import Path

ROOT=Path(__file__).resolve().parent/'data/picnic-reference-v1'
class Text(HTMLParser):
    def __init__(self): super().__init__(); self.parts=[]; self.skip=0
    def handle_starttag(self,tag,attrs):
        if tag in ('script','style'): self.skip+=1
        if tag in ('p','br','div','tr','h1','h2'): self.parts.append('\n')
    def handle_endtag(self,tag):
        if tag in ('script','style'): self.skip=max(0,self.skip-1)
        if tag in ('p','div','tr','h1','h2'): self.parts.append('\n')
    def handle_data(self,data):
        if not self.skip:self.parts.append(data)

root=ROOT/'sources';root.mkdir(parents=True,exist_ok=True)
manifest={'work':'Пикник на обочине','authors':['Аркадий Стругацкий','Борис Стругацкий'],
 'scope':'Novel only; no film, games or fan-fiction.',
 'access':'Public reading pages on rusf.ru. Public accessibility is not an open training/redistribution license; no such license was established.',
 'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sources':[]}
for i in range(1,6):
    url=f'https://www.rusf.ru/abs/books/pno{i:02d}.htm'
    request=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (local reference dataset research)'})
    with urllib.request.urlopen(request,timeout=60) as response:
        raw=response.read()
    m=re.search(rb'charset\s*=\s*["\']?([\w-]+)',raw[:5000],re.I)
    encoding=m.group(1).decode() if m else 'windows-1251'
    page=raw.decode(encoding,errors='strict')
    parser=Text();parser.feed(page)
    paragraphs=[re.sub(r'\s+',' ',p).strip() for p in ''.join(parser.parts).split('\n')]
    paragraphs=[p for p in paragraphs if p]
    sid='prologue' if i==1 else f'chapter-{i-1}'
    (root/f'{sid}.html').write_bytes(raw)
    (root/f'{sid}.json').write_text(json.dumps({'id':sid,'url':url,'paragraphs':[
        {'id':f'{sid}-p{n:04d}','text':p} for n,p in enumerate(paragraphs,1)]},ensure_ascii=False,indent=2),encoding='utf-8')
    manifest['sources'].append({'id':sid,'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'encoding':encoding,'paragraphs':len(paragraphs)})
(ROOT/'source-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
