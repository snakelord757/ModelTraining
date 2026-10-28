from pathlib import Path
import json,hashlib
from huggingface_hub import HfApi,snapshot_download
root=Path(__file__).resolve().parent
repo='intfloat/multilingual-e5-small'
info=HfApi().model_info(repo,files_metadata=True)
out=root/'models/multilingual-e5-small'
patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','README.md']
snapshot_download(repo,revision=info.sha,local_dir=out,allow_patterns=patterns,max_workers=3)
files=[]
for item in info.siblings:
    if item.rfilename not in patterns:continue
    p=out/item.rfilename
    assert p.stat().st_size==item.size
    with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    if item.lfs:assert digest==item.lfs.sha256
    files.append({'name':item.rfilename,'sha256':digest,'bytes':item.size})
(out/'download-verification.json').write_text(json.dumps({'repo':repo,'revision':info.sha,'verified':True,'files':files},indent=2))
print('ENCODER VERIFIED',info.sha,flush=True)
