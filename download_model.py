"""Download and verify the official instruction-tuned Gemma checkpoint."""
from pathlib import Path
import hashlib
import json
from huggingface_hub import HfApi, snapshot_download

repo = 'google/gemma-4-12B-it'
revision = '707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
destination = Path(__file__).resolve().parent / 'models' / 'gemma-4-12B-it'
info = HfApi().model_info(repo, revision=revision, files_metadata=True)
print(f'Downloading {repo} at {revision}', flush=True)
snapshot_download(repo_id=repo, revision=revision, local_dir=destination,
                  max_workers=4)
files = []
for item in info.siblings:
    path = destination / item.rfilename
    assert path.is_file(), f'Missing {path}'
    assert path.stat().st_size == item.size, f'Size mismatch: {path}'
    entry = {'name': item.rfilename, 'bytes': item.size}
    if item.lfs and item.lfs.sha256:
        print(f'Verifying SHA256: {item.rfilename}', flush=True)
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        assert digest == item.lfs.sha256, f'Hash mismatch: {path}'
        entry['sha256'] = digest
    files.append(entry)
report = {'repo': repo, 'revision': revision, 'files': files, 'verified': True}
(destination / 'download-verification.json').write_text(json.dumps(report, indent=2))
print('DOWNLOAD VERIFIED', flush=True)
