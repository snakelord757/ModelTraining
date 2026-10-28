"""Wait for a complete download, then verify all repository files."""
import hashlib
import json
import time
from pathlib import Path

root = Path(__file__).resolve().parent
folder = root / 'models' / 'gemma-4-12B-it'
metadata = json.loads((folder / 'source-metadata.json').read_text(encoding='utf-8-sig'))
status = root / 'download-status.json'
expected = sum(item['size'] for item in metadata['siblings'])
while True:
    actual = sum(min((folder / item['rfilename']).stat().st_size, item['size'])
                 for item in metadata['siblings'] if (folder / item['rfilename']).is_file())
    complete = all((folder / item['rfilename']).is_file()
                   and (folder / item['rfilename']).stat().st_size == item['size']
                   for item in metadata['siblings'])
    status.write_text(json.dumps({'state': 'verifying' if complete else 'downloading',
                                 'downloaded_bytes': actual, 'total_bytes': expected}, indent=2))
    if complete:
        break
    time.sleep(30)
try:
    for item in metadata['siblings']:
        path = folder / item['rfilename']
        if item.get('lfs'):
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            assert digest == item['lfs']['sha256'], f'SHA256 mismatch: {path.name}'
        else:
            content = path.read_bytes()
            digest = hashlib.sha1(f'blob {len(content)}\0'.encode() + content).hexdigest()
            assert digest == item['blobId'], f'Git blob hash mismatch: {path.name}'
    result = {'state': 'verified', 'repo': metadata['id'], 'revision': metadata['sha'],
              'total_bytes': expected, 'files': len(metadata['siblings'])}
    (folder / 'download-verification.json').write_text(json.dumps(result, indent=2))
    status.write_text(json.dumps(result, indent=2))
except Exception as exc:
    status.write_text(json.dumps({'state': 'verification_failed', 'error': str(exc)}, indent=2))
    raise
