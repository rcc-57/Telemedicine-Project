"""Own only processes started here; shutdown on Ctrl+C, GUI close or service failure."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
children = []
logs = []
stopping = False
windows_job = None


def stop_signal(*_):
    global stopping
    stopping = True


def start(arguments, env, logfile=None):
    global windows_job
    output = None
    if logfile:
        output = (ROOT / 'runtime' / logfile).open('a')
        logs.append(output)
    if os.name == 'nt' and windows_job is None:
        from telemed.windows_processes import ProcessJob
        windows_job = ProcessJob()
    options = ({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt'
               else {'start_new_session': True})
    child = subprocess.Popen(arguments, cwd=ROOT, env=env, stdout=output, stderr=output, **options)
    if windows_job:
        try:
            windows_job.assign(child)
        except OSError:
            child.kill()
            child.wait()
            raise
    children.append(child)
    return child


def stop_all():
    # Client closes devices and leaves the room before the server exits.
    for child in reversed(children):
        if child.poll() is None:
            try:
                child.send_signal(signal.CTRL_BREAK_EVENT if os.name == 'nt' else signal.SIGTERM)
            except OSError:
                child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    child.kill()
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        # Inference subprocesses share their parent's process group.
        if os.name != 'nt':
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    if windows_job:
        windows_job.close()
    for output in logs:
        output.close()


def ready_ollama(port):
    try:
        with urlopen(f'http://127.0.0.1:{port}/api/tags', timeout=1) as response:
            return json.load(response)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description='Телемедицина: единый запуск, остановка Ctrl+C')
    parser.add_argument('--client-only', action='store_true', help='Только клиент для второго участника')
    parser.add_argument('--host', default='127.0.0.1', help='IP сервера для клиента')
    parser.add_argument('--bind', default='0.0.0.0', help='Адрес сервера в режиме полного запуска')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--ollama-port', type=int, default=11435)
    parser.add_argument('--download-model', action='store_true')
    parser.add_argument('--server-only', action='store_true')
    parser.add_argument('--check', action='store_true', help='Проверить сервисы и завершиться без окна')
    args = parser.parse_args()
    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    env.update(OLLAMA_HOST=f'127.0.0.1:{args.ollama_port}', OLLAMA_MODELS=str(ROOT / 'models' / 'ollama'),
               TELEMED_OLLAMA_URL=f'http://127.0.0.1:{args.ollama_port}', OLLAMA_NO_CLOUD='1',
               HF_HOME=str(ROOT / '.cache' / 'huggingface'), MPLCONFIGDIR=str(ROOT / '.cache' / 'matplotlib'),
               PYTHONPATH=str(ROOT) + os.pathsep + str(ROOT / 'Telemedicine_Client'))
    (ROOT / 'runtime').mkdir(exist_ok=True)
    signal.signal(signal.SIGINT, stop_signal)
    signal.signal(signal.SIGTERM, stop_signal)
    try:
        if not args.client_only:
            ollama = shutil.which('ollama')
            if not ollama:
                raise RuntimeError('Ollama не установлена. Установите приложение с ollama.com.')
            # Never adopt an unrelated daemon: this launcher controls its own lifecycle.
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    probe.bind(('127.0.0.1', args.ollama_port))
                except OSError:
                    raise RuntimeError(f'Порт Ollama {args.ollama_port} занят. Закройте прежний запуск или задайте --ollama-port.')
            daemon = start([ollama, 'serve'], env, 'ollama.log')
            deadline = time.monotonic() + 25
            tags = None
            while not stopping and time.monotonic() < deadline and daemon.poll() is None:
                tags = ready_ollama(args.ollama_port)
                if tags is not None:
                    break
                time.sleep(.2)
            if stopping:
                return 0
            if tags is None:
                raise RuntimeError('Ollama не запустилась. Подробности: runtime/ollama.log')
            model = env.get('TELEMED_OLLAMA_MODEL', 'qwen3:4b')
            if args.download_model:
                print(f'Загрузка {model} в models/ollama…', flush=True)
                pull = start([ollama, 'pull', model], env)
                while pull.poll() is None and not stopping:
                    time.sleep(.2)
                return 0 if stopping else pull.returncode
            if model not in [m['name'] for m in tags.get('models', [])]:
                raise RuntimeError('Нет модели Ollama. Выполните: bash setup.sh')
            if not (ROOT / 'models/whisper/small/model.bin').is_file():
                raise RuntimeError('Нет модели Whisper. Выполните: bash setup.sh')
            # Check UDP port before launching. The server remains the source of truth.
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.bind((args.bind, args.port))
            server = start([sys.executable, '-u', 'HeadAnalysisServer.py', '--host', args.bind, '--port', str(args.port)], env, 'server.log')
            # A malformed request gets a response without creating a consultation.
            deadline = time.monotonic() + 35
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                probe.settimeout(.3)
                host = '127.0.0.1' if args.bind == '0.0.0.0' else args.bind
                ready = False
                while not stopping and server.poll() is None and time.monotonic() < deadline:
                    probe.sendto(b'{}', (host, args.port))
                    try:
                        probe.recv(4096)
                        ready = True
                        break
                    except socket.timeout:
                        pass
            if stopping:
                return 0
            if not ready:
                raise RuntimeError('Сервер не запустился. Подробности: runtime/server.log')
            print(f'Сервер UDP :{args.port} • локальная Ollama :{args.ollama_port} готовы.', flush=True)
        if args.check:
            print('Проверка запуска пройдена. Завершаем процессы.', flush=True)
            return 0
        gui = None
        if not args.server_only:
            gui = start([sys.executable, '-m', 'telemed.desktop', '--host', args.host, '--port', str(args.port)], env)
        print('Система запущена. Завершение: Ctrl+C' + (' или закрытие окна.' if gui else '.'), flush=True)
        while not stopping:
            if gui and gui.poll() is not None:
                return gui.returncode
            failed = next((child for child in children if child.poll() is not None), None)
            if failed:
                raise RuntimeError('Один из сервисов завершился. См. runtime/*.log')
            time.sleep(.2)
        return 0
    except (RuntimeError, OSError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 1
    finally:
        print('Останавливаем систему…', flush=True)
        stop_all()
        print('Все процессы этого запуска завершены.', flush=True)


if __name__ == '__main__':
    sys.exit(main())
