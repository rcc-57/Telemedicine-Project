"""UDP-медиапротокол для локальной сети; только стандартная библиотека."""

import base64
import binascii
import json
import logging
import secrets
import socket
import threading
import time
import uuid
from collections import deque

PREFIX = b"MEDIA1 "
CHUNK = 768
MAX_PARTS = 160
SAMPLE_RATE = 16000
AUDIO_FRAMES = 320  # 20 мс, mono PCM int16
AUDIO_BYTES = AUDIO_FRAMES * 2
PEER_TIMEOUT = 4


def pack(value):
    return PREFIX + json.dumps(value, separators=(",", ":")).encode()


def unpack(packet):
    if not packet.startswith(PREFIX) or len(packet) > 1800:
        raise ValueError("Неверный медиапакет")
    value = json.loads(packet[len(PREFIX):])
    if not isinstance(value, dict):
        raise ValueError("Неверный медиапакет")
    return value


def valid_id(value):
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def payload(message):
    kind = message.get("kind")
    sequence = message.get("seq")
    if kind not in ("video", "audio") or type(sequence) is not int or not 0 <= sequence < 2**53:
        raise ValueError("Неверная последовательность")
    data = base64.b64decode(message.get("data", ""), validate=True)
    if kind == "audio":
        if len(data) != AUDIO_BYTES:
            raise ValueError("Неверный аудиоблок")
    else:
        index, total = message.get("index"), message.get("total")
        if (type(index) is not int or type(total) is not int
                or not 1 <= total <= MAX_PARTS or not 0 <= index < total
                or not 1 <= len(data) <= CHUNK):
            raise ValueError("Неверная часть кадра")
    return data


def relay(rooms, packet, address, now=None, analysis=None):
    """Возвращает пакеты для отправки; камера/микрофон на сервере не нужны."""
    now = time.monotonic() if now is None else now
    try:
        message = unpack(packet)
        code, token, sid = message.get("room"), message.get("token"), message.get("sid")
        if not isinstance(code, str) or not isinstance(token, str) or not valid_id(sid):
            return []
        room = rooms.get(code)
        if room is None or now - room["created"] >= 3600:
            return [(pack({"kind": "error", "error": "Комната закрыта или срок её жизни истёк"}), address)]
        role = next((r for r in ("doctor", "patient") if room[r] is not None
                     and secrets.compare_digest(token.encode(), room[r].encode())), None)
        if role is None:
            return [(pack({"kind": "error", "error": "Участие в комнате завершено"}), address)]
        peers = room.setdefault("media", {})
        other = "patient" if role == "doctor" else "doctor"
        peer = peers.get(other)
        if peer and (room[other] is None or now - peer["seen"] > PEER_TIMEOUT):
            peer = None
        kind = message.get("kind")
        own = peers.get(role)
        if kind == "hello":
            camera, microphone = message.get("camera"), message.get("microphone")
            if type(camera) is not bool or type(microphone) is not bool:
                return []
            # Активный звонок нельзя перехватить другим медиасокетом с тем же токеном.
            if own and now - own["seen"] <= PEER_TIMEOUT and (own["address"], own["sid"]) != (address, sid):
                return [(pack({"kind": "error", "error": "Звонок уже открыт в другом окне"}), address)]
            if own is None or (own["camera"], own["microphone"], own["sid"]) != (camera, microphone, sid):
                logging.getLogger(__name__).info("%s:%s | комната %s | %s | камера %s, микрофон %s",
                    *address, code, role, "вкл" if camera else "выкл", "вкл" if microphone else "выкл")
            peers[role] = {"address": address, "sid": sid, "seen": now,
                           "camera": camera, "microphone": microphone}
            if analysis is not None:
                analysis.control(code, role, room, message, now)
            return [(pack({"kind": "state", "peer": bool(peer),
                           "sid": peer["sid"] if peer else None,
                           "camera": bool(peer and peer["camera"]),
                           "microphone": bool(peer and peer["microphone"]),
                           "analysis": analysis.result(code, room, now) if analysis is not None else {"state": "disabled"}}), address)]
        if own is None or own["address"] != address or own["sid"] != sid or now - own["seen"] > PEER_TIMEOUT:
            return []
        if kind == "bye":
            if analysis is not None:
                analysis.control(code, role, room, message, now)
            del peers[role]
            logging.getLogger(__name__).info("%s:%s | комната %s | %s завершил звонок", *address, code, role)
            return []
        payload(message)
        if not own["camera" if kind == "video" else "microphone"]:
            return []
        if analysis is not None and kind == "video" and role == "patient":
            analysis.video(code, sid, message, now)
        if peer is None:
            return []
        # Токен отправителя получателю не передаём.
        clean = {key: message[key] for key in ("kind", "sid", "seq", "data")}
        if kind == "video":
            clean.update(index=message["index"], total=message["total"])
        return [(pack(clean), peer["address"])]
    except (ValueError, TypeError, KeyError, UnicodeError, binascii.Error, RecursionError):
        return []


class VideoAssembler:
    """Собирает новые кадры; старые, неполные и повторные пакеты отбрасывает."""

    def __init__(self):
        self.frames = {}
        self.last = -1

    def add(self, message, now=None):
        now = time.monotonic() if now is None else now
        data = payload(message)
        sequence = message["seq"]
        self.frames = {key: value for key, value in self.frames.items()
                       if key > self.last and now - value[0] < 0.5}
        if sequence <= self.last:
            return None
        if sequence not in self.frames:
            if len(self.frames) >= 4:
                del self.frames[min(self.frames)]
            self.frames[sequence] = (now, message["total"], {})
        _, total, parts = self.frames[sequence]
        if total != message["total"]:
            return None
        parts[message["index"]] = data
        if len(parts) == total:
            self.last = sequence
            result = b"".join(parts[i] for i in range(total))
            self.frames = {key: value for key, value in self.frames.items() if key > sequence}
            return result
        return None


class MediaConnection:
    """Отдельный сокет звонка: не смешивает медиаданные с ответами чата."""

    def __init__(self, server, room, token, audio_observer=None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
        self.sock.connect(server)
        self.sock.settimeout(0.01)
        self.identity = {"room": room, "token": token, "sid": uuid.uuid4().hex}
        self.camera = False
        self.microphone = False
        self.speaker = True
        self.lock = threading.RLock()
        self.stopped = threading.Event()
        self.out_audio = deque(maxlen=5)
        self.in_audio = {}
        self.last_audio = -1
        self.video = None
        self.video_time = 0
        self.assembler = VideoAssembler()
        self.peer_sid = None
        self.peer_camera = False
        self.peer_microphone = False
        self.last_state = 0
        self.status = "Подключение к серверу…"
        self.fatal = False
        self.video_seq = self.audio_seq = 0
        self.next_hello = 0
        self.analysis = {"state": "waiting"}
        self.analysis_time = 0
        self.calibration = None
        self.audio_observer = audio_observer
        self.observed_audio = set()
        self.thread = threading.Thread(target=self._run, daemon=True, name="UDP media")
        self.thread.start()

    def _send(self, message):
        self.sock.send(pack({**self.identity, **message}))

    def calibrate(self):
        with self.lock:
            self.calibration = uuid.uuid4().hex
            self.analysis = {"state": "calibrating"}
            self.next_hello = 0

    def controls(self, camera=None, microphone=None, speaker=None):
        with self.lock:
            if camera is not None:
                self.camera = camera
            if microphone is not None:
                self.microphone = microphone
                if not microphone:
                    self.out_audio.clear()
            if speaker is not None:
                self.speaker = speaker
                self.in_audio.clear()
            self.next_hello = 0

    def send_video(self, jpeg):
        total = (len(jpeg) + CHUNK - 1) // CHUNK
        if not 1 <= total <= MAX_PARTS:
            return
        with self.lock:
            self.video_seq += 1
            sequence = self.video_seq
        for index in range(total):
            with self.lock:
                if not self.camera or self.stopped.is_set():
                    return
                self._send({"kind": "video", "seq": sequence, "index": index, "total": total,
                            "data": base64.b64encode(jpeg[index * CHUNK:(index + 1) * CHUNK]).decode()})

    def capture_audio(self, data):
        with self.lock:
            if self.microphone and len(data) == AUDIO_BYTES:
                self.out_audio.append(data)

    def playback_audio(self):
        with self.lock:
            if not self.speaker or not self.peer_microphone or not self.in_audio:
                return bytes(AUDIO_BYTES)
            sequence = min(self.in_audio)
            self.last_audio = sequence
            return self.in_audio.pop(sequence)

    def _receive(self, message):
        kind = message.get("kind")
        if kind == "error":
            self.status = str(message.get("error", "Ошибка звонка"))
            self.fatal = True
            self.controls(camera=False, microphone=False)
            self.stopped.set()
        elif kind == "state":
            self.last_state = time.monotonic()
            result = message.get("analysis", {"state": "disabled"})
            if isinstance(result, dict):
                self.analysis = result
                self.analysis_time = self.last_state
            sid = message.get("sid")
            if sid != self.peer_sid:
                self.peer_sid = sid
                self.assembler = VideoAssembler()
                self.in_audio.clear()
                self.last_audio = -1
                self.observed_audio.clear()
                self.video = None
            self.peer_camera = bool(message.get("camera"))
            self.peer_microphone = bool(message.get("microphone"))
            if not self.peer_camera:
                self.video = None
            if not self.peer_microphone:
                self.in_audio.clear()
            self.status = "Собеседник подключён" if message.get("peer") else "Ожидание звонка собеседника…"
        elif message.get("sid") == self.peer_sid and self.peer_sid is not None:
            if kind == "video" and self.peer_camera:
                jpeg = self.assembler.add(message)
                if jpeg is not None:
                    self.video, self.video_time = jpeg, time.monotonic()
            elif kind == "audio" and self.peer_microphone:
                data = payload(message)
                sequence = message["seq"]
                if self.audio_observer and sequence not in self.observed_audio:
                    self.observed_audio.add(sequence)
                    if len(self.observed_audio) > 1000:
                        self.observed_audio = {n for n in self.observed_audio if n > sequence - 500}
                    self.audio_observer(sequence, data)
                if self.speaker and sequence > self.last_audio:
                    self.in_audio[sequence] = data
                    if len(self.in_audio) > 6:
                        self.last_audio = min(self.in_audio)
                        del self.in_audio[self.last_audio]

    def _run(self):
        while not self.stopped.is_set():
            try:
                with self.lock:
                    now = time.monotonic()
                    if now >= self.next_hello:
                        self._send({"kind": "hello", "camera": self.camera, "microphone": self.microphone,
                                    "calibration": self.calibration})
                        self.next_hello = now + 0.7
                    if self.last_state and now - self.last_state > PEER_TIMEOUT:
                        self.status = "Нет ответа сервера…"
                        self.video = None
                        self.in_audio.clear()
                        self.peer_camera = self.peer_microphone = False
                        self.analysis = {"state": "stale"}
                    while self.out_audio:
                        data = self.out_audio.popleft()
                        if self.microphone:
                            self.audio_seq += 1
                            self._send({"kind": "audio", "seq": self.audio_seq,
                                        "data": base64.b64encode(data).decode()})
                packet = self.sock.recv(2048)
                with self.lock:
                    self._receive(unpack(packet))
            except socket.timeout:
                pass
            except (ValueError, TypeError, KeyError, binascii.Error):
                pass
            except OSError:
                if not self.stopped.is_set():
                    self.status = "Ошибка сети: проверьте сервер и подключение"
                    self.stopped.wait(0.2)

    def close(self):
        self.controls(camera=False, microphone=False)
        self.stopped.set()
        self.thread.join(timeout=1)
        try:
            for _ in range(3):
                self._send({"kind": "bye"})
        except OSError:
            pass
        self.sock.close()
