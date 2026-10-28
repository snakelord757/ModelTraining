"""Download a pinned snapshot, then verify every selected model file."""
import hashlib
import json
import os
import subprocess
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import HfApi, hf_hub_download
from model_settings import MODEL_ID, MODEL_PATH

MODEL_PATH.mkdir(parents=True, exist_ok=True)
api = HfApi()
metadata_path = MODEL_PATH / 'source-metadata.json'
if metadata_path.exists():
    revision = json.loads(metadata_path.read_text())['revision']
else:
    revision = api.model_info(MODEL_ID).sha
info = api.model_info(MODEL_ID, revision=revision, files_metadata=True)
metadata_path.write_text(json.dumps({'repo': MODEL_ID, 'revision': revision}, indent=2))
verified = []
for item in info.siblings:
    if not item.rfilename.endswith(('.json', '.safetensors', '.jinja', '.txt', '.md')):
        continue
    print(f'Downloading {item.rfilename}', flush=True)
    if item.rfilename.endswith('.safetensors'):
        path = MODEL_PATH / item.rfilename
        if not path.exists() or path.stat().st_size != item.size:
            winpath = subprocess.check_output(['wslpath', '-w', str(path)], text=True).strip()
            subprocess.run(['/mnt/c/Windows/System32/curl.exe', '-L', '--fail', '--retry', '5',
                '--connect-timeout', '30', '--speed-time', '90', '--speed-limit', '1024',
                '-C', '-', '-o', winpath,
                f'https://huggingface.co/{MODEL_ID}/resolve/{revision}/{item.rfilename}?download=true'], check=True)
    else:
        path = hf_hub_download(MODEL_ID, item.rfilename, revision=revision, local_dir=MODEL_PATH)
    h = hashlib.sha256() if item.lfs else hashlib.sha1()
    if not item.lfs:
        h.update(f'blob {item.size}\0'.encode())
    with open(path, 'rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            h.update(chunk)
    expected = item.lfs.sha256 if item.lfs else item.blob_id
    if h.hexdigest() != expected:
        raise RuntimeError(f'Integrity check failed: {item.rfilename}')
    verified.append({'file': item.rfilename, 'bytes': item.size, 'hash': h.hexdigest()})
    print(f'Verified {item.rfilename}', flush=True)
(MODEL_PATH / 'download-verification.json').write_text(json.dumps({
    'repo': MODEL_ID, 'revision': revision, 'verified': True, 'files': verified}, indent=2))
print('DOWNLOAD VERIFIED', flush=True)
