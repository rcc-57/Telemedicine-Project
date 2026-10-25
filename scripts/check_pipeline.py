"""Exercise real Whisper + Ollama through cancellable multiprocessing IPC."""
import sys
from pathlib import Path
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from telemed.speech import SpeechPipeline
from faster_whisper.audio import decode_audio


def main():
    samples = decode_audio(str(ROOT / 'runtime/test-speech.aiff'), sampling_rate=16000)
    pcm = (samples * 32767).astype('int16').tobytes()
    pipeline = SpeechPipeline()
    try:
        deadline = time.monotonic()+40
        ready = False
        while time.monotonic() < deadline and not ready:
            for item in pipeline.poll():
                assert item['kind'] != 'error', item
                ready |= item.get('text', '').startswith('Готово')
            time.sleep(.1)
        assert ready, 'Model did not become ready'
        for index in range(0, len(pcm), 640):
            pipeline.feed('patient', index//640, pcm[index:index+640].ljust(640, b'\0'))
        deadline = time.monotonic()+90
        rows = []
        while time.monotonic() < deadline:
            for item in pipeline.poll():
                assert item['kind'] != 'error', item
                if item['kind'] == 'transcript':
                    rows.append(item['text'])
            if len(rows) >= (len(pcm)//640 + 749)//750:
                break
            time.sleep(.2)
        result = ' '.join(rows)
        print(result)
        assert rows and 'Петров' not in result and 'Иван' not in result, result
        assert '500' in result, result
        print('PASS: real audio → Whisper → Ollama → masked IPC transcript')
    finally:
        pipeline.close()
        assert not pipeline.process.is_alive()
        print('PASS: inference process terminated')


if __name__ == '__main__':
    main()
