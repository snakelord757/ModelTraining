import json,hashlib,re
from pathlib import Path
from pypdf import PdfReader
import pdfplumber
ROOT=Path(__file__).resolve().parent/'data/picnic-pdf-reference-v1'
SOURCE=Path('C:/Users/Admin/Desktop/1971-Пикник-на-обочине-без-илл.pdf')
ROOT.mkdir(parents=True,exist_ok=True)
reader=PdfReader(str(SOURCE))
pages=[]
for i,page in enumerate(reader.pages,1):
    text=page.extract_text() or ''
    pages.append({'id':f'pdf-p{i:03d}','page':i,'text':text})
(ROOT/'pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'source-manifest.json').write_text(json.dumps({'source_path':str(SOURCE),'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
 'page_count':len(pages),'source_policy':'User-provided PDF only; online draft is not an evidence source for this dataset.',
 'page_numbering':'1-based PDF page index, not printed folio'},ensure_ascii=False,indent=2),encoding='utf-8')
with pdfplumber.open(str(SOURCE)) as doc:
    for i in [0,len(doc.pages)-1]:
        doc.pages[i].to_image(resolution=80).save(str(ROOT/f'preview-{i+1}.png'))
print('Pages:',len(pages),'Characters:',sum(len(p['text']) for p in pages))
print(pages[0]['text'][:1400])
