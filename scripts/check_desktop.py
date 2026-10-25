"""Real Qt widgets + two clients, synthetic audio/video, no camera access."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import threading
import time
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication
from telemed.desktop import Window, STYLE
from Lab_Work_two import UDPServer

app = QApplication([])
app.setStyle('Fusion')
app.setStyleSheet(STYLE)
server = UDPServer(('127.0.0.1', 0))
thread = threading.Thread(target=server.serve_forever, daemon=True)
thread.start()
port = server.server_address[1]


def until(predicate, seconds=8):
    deadline = time.monotonic()+seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.02)
    assert predicate(), 'Timed out waiting for UI state'


with patch('telemed.desktop.CallMedia.speaker'):
    doctor = Window(port=port)
    patient = Window(port=port)
    doctor.show()
    patient.show()
    app.processEvents()
    doctor.grab().save(str(ROOT / 'runtime' / 'desktop-lobby.png'))
    try:
        doctor.join(None)
        until(lambda: doctor.media is not None)
        patient.join(doctor.worker.client.room)
        until(lambda: patient.media is not None)
        doctor.message_edit.setText('Добрый день! Как вы себя чувствуете?')
        doctor.send_text()
        until(lambda: 'Добрый день!' in patient.chat.toPlainText())
        patient.message_edit.setText('Здравствуйте. Боль стала меньше.')
        patient.send_text()
        until(lambda: 'Боль стала меньше' in doctor.chat.toPlainText())
        until(lambda: doctor.transcript_epoch == patient.transcript_epoch and doctor.transcript_epoch >= 1)
        assert not doctor.ai_button.isHidden() and patient.ai_button.isHidden()
        assert patient.export_button.isHidden()
        doctor.publish_transcript({'kind': 'status', 'text': 'Транскрибация запущена врачом'})
        doctor.publish_transcript({'kind': 'transcript', 'role': 'patient', 'captured': time.time(),
                                   'text': '[ИМЯ_1]: принимаю 500 мг'})
        until(lambda: '500 мг' in patient.transcript_view.toPlainText())
        until(lambda: 'запущена врачом' in patient.ai_status.text())
        assert patient.transcripts == doctor.transcripts
        patient.toggle_ai(True)
        assert patient.media.pipeline is None
        doctor.publish_transcript({'kind': 'status', 'text': 'Транскрибация остановлена врачом'})
        until(lambda: 'остановлена врачом' in patient.ai_status.text())
        assert not doctor.mic_button.isChecked()
        assert not doctor.camera_button.isChecked()
        doctor.timer.stop()
        frame = np.zeros((360, 480, 3), dtype=np.uint8)
        frame[:] = (49, 71, 88)
        doctor.local_tile.set_frame(frame)
        doctor.status.setText('Проверка интерфейса • два клиента подключены')
        app.processEvents()
        doctor.grab().save(str(ROOT / 'runtime' / 'desktop-preview.png'))
        patient.leave()
        until(lambda: server.application.rooms[doctor.worker.client.room]['patient'] is None)
        doctor.leave()
        until(lambda: not server.application.rooms)
        print('PASS: two GUI clients, room lifecycle, bidirectional chat, safe device defaults, screenshot')
    finally:
        doctor.close()
        patient.close()
        server.shutdown()
        server.server_close()
