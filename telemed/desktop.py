"""Desktop consultation room. All widgets are updated only on the Qt thread."""
import argparse
from collections import deque
from datetime import datetime
import html
from pathlib import Path
import queue
import signal
import sys
import threading
import time

import cv2
import numpy as np
import sounddevice as sd
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPixmap, QFont
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QLineEdit, QComboBox, QVBoxLayout, QHBoxLayout, QGridLayout, QTabWidget,
    QTextBrowser, QFileDialog, QSplitter, QSizePolicy)
from Telemedicine_Client.Client import Client, ServerError
from Telemedicine_Client.media import MediaConnection, AUDIO_BYTES, AUDIO_FRAMES, SAMPLE_RATE
from Telemedicine_Client.VideoCall import analysis_text
from telemed.speech import SpeechPipeline, ROOT

STYLE = '''
QWidget { color: #e8edf5; font-family: "Avenir Next", "Segoe UI", sans-serif; font-size: 13px; }
QMainWindow, QWidget#shell { background: #141a24; }
QFrame#sidebar, QFrame#panel { background: #1c2430; border-radius: 22px; }
QFrame#rail { background: #18202b; border-radius: 18px; }
QLabel#muted { color: #8996ab; font-size: 12px; }
QLabel#title { font-size: 23px; font-weight: 600; }
QLabel#brand { font-size: 17px; font-weight: 700; color: #8adbad; }
QLabel#pill { background: #223b32; color: #8cddb0; border-radius: 12px; padding: 7px 12px; }
QPushButton { background: #293342; border: 1px solid #344052; border-radius: 12px; padding: 11px 15px; }
QPushButton:hover { background: #354258; }
QPushButton:checked { background: #294d41; color: #a0edc1; border: 1px solid #4c8c70; }
QPushButton:disabled { color: #657184; background: #202834; }
QPushButton#primary { background: #74ca9b; color: #10291e; border: none; font-weight: 600; }
QPushButton#primary:hover { background: #93e0b6; }
QPushButton#hangup { background: #e24767; color: white; border: none; }
QLineEdit, QComboBox { background: #141c27; border: 1px solid #344052; border-radius: 10px; padding: 10px; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView { background: #202a39; selection-background-color: #365b4a; }
QTextBrowser { background: #1c2430; border: none; padding: 10px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: #1c2430; color: #8d9bb0; padding: 13px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #8ee3b1; border-bottom: 2px solid #8ee3b1; }
QSplitter::handle { background: transparent; width: 12px; }
QScrollBar:vertical { background: transparent; width: 6px; }
QScrollBar::handle:vertical { background: #445065; border-radius: 3px; min-height: 24px; }
'''
STYLE += 'QComboBox::down-arrow { image: url("' + (ROOT / 'telemed/assets/chevron.svg').as_posix() + '"); width: 12px; height: 8px; }'


def label(text, name=None):
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(True)
    return widget


def button(text, callback, name=None, checkable=False):
    widget = QPushButton(text)
    if name:
        widget.setObjectName(name)
    widget.setFixedHeight(44)
    widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    widget.setCheckable(checkable)
    widget.clicked.connect(callback)
    return widget


class VideoTile(QWidget):
    def __init__(self, title, initials, tint):
        super().__init__()
        self.title, self.initials, self.tint = title, initials, QColor(tint)
        self.frame = None
        self.note = 'Камера выключена'
        self.setMinimumSize(220, 270)

    def set_frame(self, array):
        if array is None:
            self.frame = None
        else:
            h, w, _ = array.shape
            self.frame = QImage(array.data, w, h, array.strides[0], QImage.Format.Format_RGB888).copy()
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(1, 1, -1, -1)
        path = QPainterPath()
        path.addRoundedRect(rect.toRectF(), 24, 24)
        p.setClipPath(path)
        p.fillRect(rect, self.tint)
        if self.frame:
            pix = QPixmap.fromImage(self.frame).scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                     Qt.TransformationMode.SmoothTransformation)
            p.drawPixmap((self.width()-pix.width())//2, (self.height()-pix.height())//2, pix)
        else:
            p.setPen(QColor('#dce4f5'))
            p.setFont(QFont('Avenir Next', 42, QFont.Weight.Medium))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.initials)
            p.setFont(QFont('Avenir Next', 11))
            p.setPen(QColor('#b5bed1'))
            p.drawText(rect.adjusted(12, 95, -12, 0), Qt.AlignmentFlag.AlignCenter, self.note)
        p.fillRect(0, self.height()-50, self.width(), 50, QColor(13, 20, 30, 190))
        p.setPen(QColor('#eef2f9'))
        p.setFont(QFont('Avenir Next', 12))
        p.drawText(18, self.height()-20, self.title)
        p.end()


class Bridge(QObject):
    result = Signal(str, object)


class ClientWorker(threading.Thread):
    def __init__(self, bridge, host, port, code):
        super().__init__(daemon=True, name='Room requests')
        self.bridge, self.host, self.port, self.code = bridge, host, port, code
        self.jobs = queue.Queue()
        self.stop_event = threading.Event()
        self.client = None
        self.transcript_epoch = -1
        self.transcript_after = 0
        self.retry_transcript = None

    def run(self):
        try:
            self.client = Client(self.host, self.port)
            self.client.enter(self.code)
            self.bridge.result.emit('joined', (self.client.room, self.client.token, self.client.role))
            while not self.stop_event.is_set():
                if self.retry_transcript is not None:
                    kind, value = 'transcript', self.retry_transcript
                    self.stop_event.wait(.3)
                else:
                    try:
                        kind, value = self.jobs.get(timeout=.3)
                    except queue.Empty:
                        kind, value = 'poll', None
                try:
                    if kind == 'text':
                        self.client.send_message(value)
                        self.bridge.result.emit('sent', value)
                    elif kind == 'image':
                        self.client.send_image(value)
                        self.bridge.result.emit('sent', 'Изображение отправлено')
                    elif kind == 'transcript':
                        self.retry_transcript = value
                        self.client.request(self.client.room_path() + '/transcript', body=value)
                        self.retry_transcript = None
                    # Never change an unresolved POST: let user retry exactly that action.
                    if self.client.pending is None:
                        for message in self.client.receive_messages(ROOT / 'received_images'):
                            self.bridge.result.emit('message', message)
                        snapshot = self.client.request(self.client.room_path() +
                            f'/transcript?after={self.transcript_after}&epoch={self.transcript_epoch}', method='GET')
                        changed = self.transcript_epoch != snapshot['epoch']
                        self.transcript_epoch = snapshot['epoch']
                        self.transcript_after = snapshot['next_after']
                        self.bridge.result.emit('transcript_snapshot', {**snapshot, 'changed': changed})
                except (OSError, ValueError, ServerError) as error:
                    if isinstance(error, ServerError):
                        self.retry_transcript = None
                    self.bridge.result.emit('error', str(error))
                    if isinstance(error, ServerError) and error.status in (401, 404):
                        self.bridge.result.emit('ended', None)
                        break
        except Exception as error:
            self.bridge.result.emit('join_error', str(error))
        finally:
            if self.client:
                try:
                    if self.client.room and self.client.pending is None:
                        self.client.leave()
                except Exception:
                    pass
                self.client.close()


class CallMedia:
    def __init__(self, host, port, room, token, role, camera, input_device, output_device):
        self.role, self.camera_index = role, camera
        self.input_device, self.output_device = input_device, output_device
        self.pipeline = None
        self.local_sequence = 0
        self.connection = MediaConnection((host, port), room, token, audio_observer=self.remote_audio)
        self.input_stream = self.output_stream = None
        self.camera_enabled = threading.Event()
        self.stop = threading.Event()
        self.preview = None
        self.notices = deque(maxlen=10)
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()

    def remote_audio(self, sequence, pcm):
        pipeline = self.pipeline
        if pipeline:
            pipeline.feed('patient' if self.role == 'doctor' else 'doctor', sequence, pcm)

    def record(self, data, frames, timing, status):
        if not self.stop.is_set() and frames == AUDIO_FRAMES:
            pcm = bytes(data)
            self.connection.capture_audio(pcm)
            self.local_sequence += 1
            pipeline = self.pipeline
            if pipeline:
                pipeline.feed(self.role, self.local_sequence, pcm)

    def playback(self, out, frames, timing, status):
        data = self.connection.playback_audio()
        out[:] = data if len(out) == len(data) else bytes(len(out))

    def microphone(self, enabled):
        if self.input_stream:
            self.connection.controls(microphone=False)
            self.input_stream.abort()
            self.input_stream.close()
            self.input_stream = None
        if enabled:
            stream = sd.RawInputStream(samplerate=SAMPLE_RATE, blocksize=AUDIO_FRAMES, channels=1,
                dtype='int16', device=self.input_device, callback=self.record)
            try:
                stream.start()
            except Exception:
                stream.close()
                raise
            self.input_stream = stream
            self.connection.controls(microphone=True)

    def speaker(self, enabled):
        if self.output_stream:
            self.output_stream.abort()
            self.output_stream.close()
            self.output_stream = None
        self.connection.controls(speaker=False)
        if enabled:
            stream = sd.RawOutputStream(samplerate=SAMPLE_RATE, blocksize=AUDIO_FRAMES, channels=1,
                dtype='int16', device=self.output_device, callback=self.playback)
            try:
                stream.start()
            except Exception:
                stream.close()
                raise
            self.output_stream = stream
            self.connection.controls(speaker=True)

    def camera_loop(self):
        capture = None
        while not self.stop.is_set():
            if not self.camera_enabled.is_set():
                if capture:
                    capture.release()
                    capture = None
                self.preview = None
                self.stop.wait(.05)
                continue
            try:
                if capture is None:
                    capture = cv2.VideoCapture(self.camera_index)
                    if not capture.isOpened():
                        raise RuntimeError('Камера недоступна. Проверьте разрешение macOS и номер устройства.')
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('Нет изображения с камеры.')
                frame = cv2.resize(frame, (480, 360))
                self.preview = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                ok, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 55])
                if ok:
                    self.connection.send_video(jpeg.tobytes())
                self.stop.wait(1/12)
            except Exception as error:
                self.notices.append(str(error))
                self.camera_enabled.clear()
                self.connection.controls(camera=False)
        if capture:
            capture.release()

    def close(self):
        self.stop.set()
        self.camera_enabled.clear()
        pipeline, self.pipeline = self.pipeline, None
        if pipeline:
            pipeline.close()
        for stream in (self.input_stream, self.output_stream):
            if stream:
                stream.abort()
                stream.close()
        self.camera_thread.join(timeout=2)
        self.connection.close()


class Window(QMainWindow):
    def __init__(self, host='127.0.0.1', port=5000):
        super().__init__()
        self.setWindowTitle('Связь • телемедицинская консультация')
        self.resize(1320, 850)
        self.setMinimumSize(1080, 740)
        self.host, self.port = host, port
        self.worker = self.media = None
        self.bridge = Bridge()
        self.bridge.result.connect(self.on_result)
        self.generation = 0
        self.transcripts = []
        self.session_role = None
        self.transcript_epoch = -1
        self.started = None
        self.last_jpeg = None
        shell = QWidget()
        shell.setObjectName('shell')
        self.setCentralWidget(shell)
        outer = QHBoxLayout(shell)
        outer.setContentsMargins(22, 22, 22, 22)
        outer.setSpacing(20)
        rail = QFrame()
        rail.setObjectName('rail')
        rail.setFixedWidth(62)
        rl = QVBoxLayout(rail)
        rl.addWidget(label('✦', 'brand'), alignment=Qt.AlignmentFlag.AlignHCenter)
        rl.addSpacing(40)
        rl.addWidget(label('◉', 'pill'))
        rl.addStretch()
        rl.addWidget(label('●', 'brand'), alignment=Qt.AlignmentFlag.AlignHCenter)
        outer.addWidget(rail)
        split = QSplitter()
        split.setHandleWidth(18)
        split.setChildrenCollapsible(False)
        outer.addWidget(split, 1)
        main = QWidget()
        ml = QVBoxLayout(main)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(16)
        heading = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label('СВЯЗЬ / TELEMED', 'brand'))
        titles.addWidget(label('Пространство консультации', 'title'))
        heading.addLayout(titles)
        heading.setStretch(0, 1)
        heading.addStretch()
        self.timer_label = label('●  Не в звонке', 'pill')
        self.timer_label.setFixedHeight(36)
        self.timer_label.setWordWrap(False)
        heading.addWidget(self.timer_label)
        ml.addLayout(heading)
        self.setup = QFrame()
        self.setup.setObjectName('panel')
        form = QGridLayout(self.setup)
        form.setContentsMargins(18, 18, 18, 18)
        self.host_edit = QLineEdit(host)
        self.host_edit.setPlaceholderText('IP сервера')
        self.code_edit = QLineEdit()
        self.code_edit.setMaxLength(6)
        self.code_edit.setPlaceholderText('Код комнаты')
        self.create_button = button('Я врач • создать', lambda: self.join(None), 'primary')
        self.join_button = button('Я пациент • войти', lambda: self.join(self.code_edit.text()))
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(12)
        for column in range(6):
            form.setColumnStretch(column, 1)
        form.addWidget(self.host_edit, 0, 0, 1, 3)
        form.addWidget(self.code_edit, 0, 3, 1, 3)
        form.addWidget(self.create_button, 1, 0, 1, 3)
        form.addWidget(self.join_button, 1, 3, 1, 3)
        self.input_combo, self.output_combo, self.camera_combo = QComboBox(), QComboBox(), QComboBox()
        self.input_combo.addItem('Микрофон: системный', None)
        self.output_combo.addItem('Динамики: системные', None)
        for i in range(4):
            self.camera_combo.addItem(f'Камера {i + 1}', i)
        try:
            for i, device in enumerate(sd.query_devices()):
                if device['max_input_channels']:
                    self.input_combo.addItem(device['name'], i)
                if device['max_output_channels']:
                    self.output_combo.addItem(device['name'], i)
        except Exception:
            pass
        for widget in (self.host_edit, self.code_edit, self.input_combo, self.output_combo, self.camera_combo):
            widget.setFixedHeight(44)
            widget.setMinimumWidth(0)
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        form.addWidget(self.input_combo, 2, 0, 1, 2)
        form.addWidget(self.output_combo, 2, 2, 1, 2)
        form.addWidget(self.camera_combo, 2, 4, 1, 2)
        ml.addWidget(self.setup)
        self.room_label = label('Создайте комнату или введите код от врача', 'muted')
        ml.addWidget(self.room_label)
        videos = QHBoxLayout()
        videos.setSpacing(16)
        self.local_tile = VideoTile('Вы', 'ВЫ', '#343047')
        self.remote_tile = VideoTile('Собеседник', '••', '#263f47')
        self.remote_tile.note = 'Ожидание подключения'
        videos.addWidget(self.local_tile)
        videos.addWidget(self.remote_tile)
        ml.addLayout(videos, 1)
        self.analysis = label('Анализ движений • ожидаем камеру пациента', 'muted')
        self.analysis.setMinimumHeight(50)
        ml.addWidget(self.analysis)
        controls = QHBoxLayout()
        self.camera_button = button('Камера', self.toggle_camera, checkable=True)
        self.mic_button = button('Микрофон', self.toggle_mic, checkable=True)
        self.speaker_button = button('Звук', self.toggle_speaker, checkable=True)
        self.end_button = button('Завершить', self.leave, 'hangup')
        self.calibrate_button = button('Калибровка', lambda: self.media and self.media.connection.calibrate())
        self.call_buttons = [self.camera_button, self.mic_button, self.speaker_button, self.end_button, self.calibrate_button]
        for b in self.call_buttons:
            b.setEnabled(False)
            controls.addWidget(b, 1)
        ml.addLayout(controls)
        self.status = label('Локальный прототип • для разговора используйте наушники', 'muted')
        ml.addWidget(self.status)
        split.addWidget(main)
        side = QFrame()
        side.setObjectName('sidebar')
        side.setMinimumWidth(310)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(18, 18, 18, 18)
        sl.addWidget(label('Диалог и заметки', 'title'))
        sl.addWidget(label('Всё важное — рядом с разговором', 'muted'))
        self.tabs = QTabWidget()
        chat = QWidget()
        cl = QVBoxLayout(chat)
        cl.setContentsMargins(0, 0, 0, 0)
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        self.chat.setOpenLinks(False)
        self.chat.setHtml('<p style="color:#8996ab">Здесь появятся сообщения участников.<br><br>Переписка доступна после входа обоих участников.</p>')
        cl.addWidget(self.chat)
        self.message_edit = QLineEdit()
        self.message_edit.setPlaceholderText('Написать сообщение…')
        self.message_edit.returnPressed.connect(self.send_text)
        cl.addWidget(self.message_edit)
        send_row = QHBoxLayout()
        send_row.addWidget(button('Изображение', self.send_image))
        send_row.addWidget(button('Отправить ↗', self.send_text, 'primary'))
        cl.addLayout(send_row)
        self.tabs.addTab(chat, 'Чат')
        transcript = QWidget()
        tl = QVBoxLayout(transcript)
        tl.setContentsMargins(0, 0, 0, 0)
        self.ai_button = button('Включить транскрибацию', self.toggle_ai, 'primary', checkable=True)
        self.ai_button.setEnabled(False)
        tl.addWidget(self.ai_button)
        self.ai_status = label('Доступно врачу после входа. Аудио и исходный текст не сохраняются.', 'muted')
        tl.addWidget(self.ai_status)
        self.transcript_view = QTextBrowser()
        tl.addWidget(self.transcript_view, 1)
        tl.addWidget(label('Проверяйте результат: распознавание и удаление личных данных могут ошибаться.', 'muted'))
        self.export_button = button('Сохранить очищенный текст', self.export)
        self.export_button.hide()
        self.ai_button.hide()
        tl.addWidget(self.export_button)
        self.tabs.addTab(transcript, 'Транскрипт')
        sl.addWidget(self.tabs, 1)
        sl.addWidget(label('Звонок: локальная сеть • UDP\nИИ: только на компьютере врача', 'muted'))
        split.addWidget(side)
        split.setSizes([860, 350])
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(60)

    def join(self, code):
        if self.worker:
            return
        if code is not None and len(code.strip()) != 6:
            self.status.setText('Введите шестизначный код комнаты.')
            return
        self.host = self.host_edit.text().strip() or '127.0.0.1'
        self.create_button.setEnabled(False)
        self.join_button.setEnabled(False)
        self.status.setText('Подключаемся…')
        self.generation += 1
        generation = self.generation
        self.bridge = Bridge(self)
        self.bridge.result.connect(lambda kind, value: self.on_result(kind, value)
                                   if generation == self.generation else None)
        self.worker = ClientWorker(self.bridge, self.host, self.port, code)
        self.worker.start()

    def on_result(self, kind, value):
        if self.worker is None:
            return
        if kind == 'joined':
            room, token, role = value
            self.session_role = role
            self.transcript_epoch = -1
            self.transcripts.clear()
            self.transcript_view.clear()
            self.chat.clear()
            try:
                self.media = CallMedia(self.host, self.port, room, token, role, self.camera_combo.currentData(),
                                       self.input_combo.currentData(), self.output_combo.currentData())
            except Exception as error:
                self.status.setText(str(error))
                self.leave()
                return
            self.started = time.monotonic()
            self.setup.hide()
            self.room_label.setText(f'Комната {room}  •  Вы — {"врач" if role == "doctor" else "пациент"}  •  Передайте код собеседнику')
            self.local_tile.title = 'Вы • ' + ('врач' if role == 'doctor' else 'пациент')
            self.remote_tile.title = 'Пациент' if role == 'doctor' else 'Врач'
            for b in self.call_buttons:
                b.setEnabled(True)
            self.ai_button.setEnabled(False)
            self.ai_button.setVisible(role == 'doctor')
            self.export_button.setVisible(role == 'doctor')
            self.transcript_view.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self.transcript_view.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse if role == 'doctor' else Qt.TextInteractionFlag.NoTextInteraction)
            self.ai_status.setText('Нажмите кнопку, чтобы включить локальную обработку речи.' if role == 'doctor'
                                   else 'Транскрибацией управляет врач на своём компьютере.')
            self.speaker_button.setChecked(True)
            self.toggle_speaker(True)
        elif kind == 'transcript_snapshot':
            if value['changed']:
                if self.media and self.media.pipeline:
                    self.toggle_ai(False)
                    self.ai_button.setChecked(False)
                self.transcript_epoch = value['epoch']
                self.transcripts.clear()
            self.transcripts.extend(value['rows'])
            if value['changed'] or value['rows']:
                self.render_transcript()
            self.ai_button.setEnabled(self.session_role == 'doctor')
            if self.session_role == 'patient':
                self.ai_status.setText(value['status'])
        elif kind == 'join_error':
            self.worker = None
            self.create_button.setEnabled(True)
            self.join_button.setEnabled(True)
            self.status.setText(value)
        elif kind == 'error':
            self.status.setText(value)
        elif kind == 'ended':
            self.leave()
            self.status.setText('Комната закрыта. Можно начать новую консультацию.')
        elif kind == 'sent':
            self.append_chat('Вы', value, '#9ce1b9')
            if value == self.message_edit.text():
                self.message_edit.clear()
        elif kind == 'message':
            who = 'Врач' if value['from'] == 'doctor' else 'Пациент'
            text = value.get('text') or ('Изображение сохранено: ' + value.get('saved_path', ''))
            self.append_chat(who, text, '#b3bfe0')
            if value.get('saved_path'):
                pix = QPixmap(value['saved_path'])
                if not pix.isNull():
                    image = pix.scaledToWidth(250, Qt.TransformationMode.SmoothTransformation).toImage()
                    self.chat.textCursor().insertImage(image)

    def append_chat(self, who, text, color):
        self.chat.append(f'<p><b style="color:{color}">{html.escape(who)}</b> '
                         f'<span style="color:#77869c">{datetime.now():%H:%M}</span><br>{html.escape(text)}</p>')

    def send_text(self):
        if self.worker and self.media and self.message_edit.text().strip():
            self.worker.jobs.put(('text', self.message_edit.text()))

    def send_image(self):
        if self.worker and self.media:
            path, _ = QFileDialog.getOpenFileName(self, 'Отправить изображение', str(ROOT), 'Изображения (*.png *.jpg *.jpeg *.webp *.gif *.bmp *.heic *.heif)')
            if path:
                self.worker.jobs.put(('image', path))
                self.status.setText('Отправка изображения…')

    def toggle_camera(self, checked):
        if self.media:
            if checked:
                self.media.camera_enabled.set()
            else:
                self.media.camera_enabled.clear()
            self.media.connection.controls(camera=checked)

    def toggle_mic(self, checked):
        if self.media:
            try:
                self.media.microphone(checked)
            except Exception:
                self.mic_button.setChecked(False)
                self.status.setText('Микрофон недоступен. Проверьте устройство и разрешения системы.')

    def toggle_speaker(self, checked):
        if self.media:
            try:
                self.media.speaker(checked)
            except Exception:
                self.speaker_button.setChecked(False)
                self.status.setText('Динамики недоступны. Проверьте выбранное устройство.')

    def toggle_ai(self, checked):
        if not self.media or self.media.role != 'doctor':
            self.ai_button.setChecked(False)
            return
        if checked:
            self.media.pipeline = SpeechPipeline()
            self.ai_status.setText('Подготовка локальных моделей…')
            self.publish_transcript({'kind': 'status', 'text': self.ai_status.text()})
            self.ai_button.setText('Остановить транскрибацию')
        else:
            pipeline, self.media.pipeline = self.media.pipeline, None
            if pipeline:
                pipeline.close()
            self.ai_status.setText('Остановлено. Незавершённые фрагменты удалены.')
            self.ai_button.setText('Включить транскрибацию')
            self.publish_transcript({'kind': 'status', 'text': 'Транскрибация остановлена врачом'})

    def tick(self):
        if not self.media:
            return
        media = self.media
        self.local_tile.set_frame(media.preview if media.camera_enabled.is_set() else None)
        with media.connection.lock:
            conn = media.connection
            jpeg = conn.video if time.monotonic()-conn.video_time < 2 else None
            status, analysis, fatal = conn.status, conn.analysis, conn.fatal
        if jpeg is not self.last_jpeg:
            array = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR) if jpeg else None
            self.remote_tile.set_frame(cv2.cvtColor(array, cv2.COLOR_BGR2RGB) if array is not None else None)
            self.last_jpeg = jpeg
        self.room_label.setText(self.room_label.text().split('  |  ')[0] + '  |  ' + status)
        self.analysis.setText(analysis_text(analysis))
        elapsed = int(time.monotonic() - self.started)
        self.timer_label.setText(f'●  {elapsed//60:02}:{elapsed%60:02}')
        while media.notices:
            self.status.setText(media.notices.popleft())
            self.camera_button.setChecked(False)
        if media.pipeline:
            for item in media.pipeline.poll():
                self.publish_transcript(item if item['kind'] == 'transcript'
                                        else {'kind': 'status', 'text': item['text']})
                if item['kind'] != 'transcript':
                    self.ai_status.setText(item['text'])
            if not media.pipeline.process.is_alive():
                media.pipeline.close()
                media.pipeline = None
                self.ai_button.setChecked(False)
                self.ai_button.setText('Повторить запуск ИИ')
        if fatal:
            self.leave()
            self.status.setText(status)

    def publish_transcript(self, item):
        if self.worker and self.session_role == 'doctor' and self.transcript_epoch >= 0:
            self.worker.jobs.put(('transcript', {**item, 'epoch': self.transcript_epoch}))

    def render_transcript(self):
        self.transcripts.sort(key=lambda row: row['captured'])
        self.transcript_view.setHtml(''.join(
            '<p><b style="color:#8edfb0">' + ('Врач' if row['role'] == 'doctor' else 'Пациент')
            + '</b> <span style="color:#8996ab">' + datetime.fromtimestamp(row['captured']).strftime('%H:%M:%S')
            + '</span><br>' + html.escape(row['text']) + '</p>' for row in self.transcripts))
        bar = self.transcript_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def export(self):
        if self.session_role != 'doctor':
            return
        if not self.transcripts:
            self.ai_status.setText('Пока нет обработанных фрагментов для сохранения.')
            return
        path, _ = QFileDialog.getSaveFileName(self, 'Сохранить транскрипт', str(ROOT / 'consultation.txt'), 'Текст (*.txt)')
        if path:
            try:
                Path(path).write_text('\n\n'.join(
                    f"{datetime.fromtimestamp(row['captured']):%H:%M:%S} • {'Врач' if row['role']=='doctor' else 'Пациент'}\n{row['text']}"
                    for row in self.transcripts), encoding='utf-8')
                self.ai_status.setText('Очищенный транскрипт сохранён.')
            except OSError:
                self.ai_status.setText('Не удалось сохранить файл. Выберите другую папку.')

    def leave(self):
        self.generation += 1
        media, self.media = self.media, None
        if media:
            media.close()
        worker, self.worker = self.worker, None
        if worker:
            worker.stop_event.set()
        self.started = None
        self.last_jpeg = None
        self.local_tile.set_frame(None)
        self.remote_tile.set_frame(None)
        self.setup.show()
        self.create_button.setEnabled(True)
        self.join_button.setEnabled(True)
        for b in self.call_buttons:
            b.setChecked(False)
            b.setEnabled(False)
        self.ai_button.setChecked(False)
        self.ai_button.setEnabled(False)
        self.ai_button.hide()
        self.ai_button.setText('Включить транскрибацию')
        self.ai_status.setText('Обработка остановлена. Готовый текст можно сохранить до следующего звонка.')
        self.timer_label.setText('●  Не в звонке')
        self.room_label.setText('Консультация завершена')
        self.analysis.setText('Анализ движений • ожидаем новую консультацию')
        return worker

    def closeEvent(self, event):
        worker = self.leave()
        if worker:
            worker.join(timeout=6.5)
        event.accept()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    app.setStyleSheet(STYLE)
    window = Window(args.host, args.port)
    signal.signal(signal.SIGINT, lambda *_: window.close())
    signal.signal(signal.SIGTERM, lambda *_: window.close())
    if hasattr(signal, 'SIGBREAK'):
        signal.signal(signal.SIGBREAK, lambda *_: window.close())
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
