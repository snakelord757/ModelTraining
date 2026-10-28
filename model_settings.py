from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_ID = 'Qwen/Qwen3-1.7B'
MODEL_PATH = ROOT / 'models/Qwen3-1.7B'
OUTPUT = 'outputs/qwen3-1.7b-literature-12k'
MAX_LENGTH = 12288
TRAIN_SAMPLES = 0
TEMPLATE_KWARGS = {'enable_thinking': False}

def require_model():
    import json
    marker = MODEL_PATH / 'download-verification.json'
    if not marker.exists():
        raise RuntimeError('Qwen download is not verified yet. Finish download_qwen.py first.')
    manifest = json.loads(marker.read_text())
    if not manifest.get('verified') or manifest.get('repo') != MODEL_ID:
        raise RuntimeError('Model verification does not match selected model')
    for entry in manifest['files']:
        file = MODEL_PATH / entry['file']
        if not file.exists() or file.stat().st_size != entry['bytes']:
            raise RuntimeError(f'Missing or incomplete model file: {file}')
    return manifest
