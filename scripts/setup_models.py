"""Download speech weights into the project, without changing global caches."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
os.environ['HF_HUB_DISABLE_XET'] = '1'
from faster_whisper.utils import download_model
print('Загрузка Whisper small (многоязычная модель)…', flush=True)
download_model('small', output_dir=str(ROOT / 'models' / 'whisper' / 'small'))
print('Whisper готов.', flush=True)
