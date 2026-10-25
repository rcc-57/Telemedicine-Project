"""Start all services and UI offscreen, send SIGINT, verify both ports are free."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
env = {**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}
with (ROOT / 'runtime/launcher-check.log').open('w') as log:
    child = subprocess.Popen([sys.executable, 'launcher.py', '--port', '5002', '--ollama-port', '11436'],
                             cwd=ROOT, env=env, stdout=log, stderr=log)
    try:
        deadline = time.monotonic()+45
        while time.monotonic() < deadline:
            assert child.poll() is None, 'Launcher failed before readiness'
            if 'Система запущена' in (ROOT / 'runtime/launcher-check.log').read_text():
                break
            time.sleep(.2)
        else:
            raise AssertionError('Readiness timeout')
        time.sleep(2)
        child.send_signal(signal.SIGINT)
        assert child.wait(timeout=20) == 0
        for port, kind in [(5002, socket.SOCK_DGRAM), (11436, socket.SOCK_STREAM)]:
            with socket.socket(socket.AF_INET, kind) as probe:
                if kind == socket.SOCK_STREAM:
                    assert probe.connect_ex(('127.0.0.1', port)) != 0, 'TCP listener still alive'
            with socket.socket(socket.AF_INET, kind) as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(('127.0.0.1', port))
        print('PASS: full launcher + Qt UI; Ctrl+C exits with code 0 and releases both ports')
    finally:
        if child.poll() is None:
            child.send_signal(signal.SIGTERM)
            child.wait(timeout=20)
