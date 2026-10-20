"""Учебный UDP-сервер ВКС. Формат сообщений и запуск описаны в README.md."""

import argparse
import base64
import binascii
import json
import hashlib
import logging
import secrets
import socket
import socketserver
import sys
import time
from collections import OrderedDict
from urllib.parse import parse_qs, urlsplit
from media import PREFIX, relay


MAX_PACKET = 60000
ROOM_TTL = 3600
logger = logging.getLogger(__name__)
IMAGE_CHUNK = 768
MAX_IMAGE = 5 * 1024 * 1024


def log_action(address, request, response, repeated=False):
    """Выводим только описание действия, без токенов и содержимого кадров."""
    request = request if isinstance(request, dict) else {}
    path = request.get("path")
    parts = path.split("?", 1)[0].strip("/").split("/") if isinstance(path, str) else []
    method = request.get("method")
    action = parts[2] if len(parts) == 3 and parts[0] == "rooms" else None
    names = {("POST", "join"): "Вход пациента",
             ("POST", "leave"): "Выход из комнаты",
             ("POST", "events"): "Отправка события",
             ("GET", "events"): "Получение событий",
             ("POST", "messages"): "Отправка сообщения",
             ("GET", "messages"): "Получение сообщений",
             ("POST", "images"): "Загрузка части картинки",
             ("GET", "images"): "Получение части картинки",
             ("POST", "frames"): "Отправка JPEG"}
    description = names.get((method, action), "Неизвестный запрос") if isinstance(method, str) else "Некорректный запрос"
    if method == "POST" and parts == ["rooms"]:
        description = "Создание комнаты врачом"
    body = response.get("body", {})
    room = body.get("room") or (parts[1] if len(parts) == 3 else None)
    # В журнал не попадает произвольный текст из path, включая управляющие символы.
    if isinstance(room, str) and len(room) == 6 and all(c in "0123456789ABCDEF" for c in room):
        description += f" | комната {room}"
    if response["status"] >= 400:
        description += " | отказ: " + response["error"]
    elif action == "leave":
        description += " | врач вышел, комната закрыта" if body["closed"] else " | пациент вышел, место свободно"
    elif action == "events" and method == "GET":
        description += f" | событий: {len(body['events'])}"
    elif action == "events" and method == "POST":
        description += f" | событие №{body['event_id']}"
    if repeated:
        description += " | повтор пакета: возвращён сохранённый ответ"
    if action == "images" and response["status"] < 400:
        if method == "GET" or not body.get("complete"):
            return
        description += " | картинка принята полностью"
    logger.info("%s:%s | %s | статус %s", address[0], address[1], description, response["status"])


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")


def analyze(frame):
    """Точка подключения алгоритма анализа JPEG, которого нет в задании."""
    return {"bytes_received": len(frame), "analysis": None,
            "message": "Кадр принят. Алгоритм анализа не подключён."}


class RequestError(Exception):
    def __init__(self, status, message):
        self.status = status
        self.message = message


class Application:
    def __init__(self):
        self.rooms = {}
        self.responses = OrderedDict()

    def handle(self, packet, address):
        """Одна UDP-датаграмма — один JSON-запрос и один JSON-ответ."""
        request_id = None
        request = None
        try:
            if len(packet) > MAX_PACKET:
                raise RequestError(413, "Сообщение должно быть не больше 60000 байт")
            request = json.loads(packet.decode("utf-8"))
            if not isinstance(request, dict):
                raise RequestError(400, "Запрос должен быть JSON-объектом")
            request_id = request.get("id")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
                request_id = None
                raise RequestError(400, "Нужен строковый id длиной от 1 до 100 символов")
            now = time.monotonic()
            self.rooms = {code: room for code, room in self.rooms.items()
                          if now - room["created"] < ROOM_TTL}
            while self.responses:
                if next(iter(self.responses.values()))[0] > now - 120:
                    break
                self.responses.popitem(last=False)
            key = (address, request_id)
            if key in self.responses:
                _, original, response = self.responses[key]
                if original != packet:
                    raise RequestError(409, "Этот id уже использован для другого запроса")
                log_action(address, request, json.loads(response), repeated=True)
                return response
            status, body = self.dispatch(request)
            response = encode({"id": request_id, "status": status, "body": body})
            self.responses[key] = (now, packet, response)
            if len(self.responses) > 1024:
                self.responses.popitem(last=False)
            log_action(address, request, json.loads(response))
            return response
        except RequestError as error:
            response = {"id": request_id, "status": error.status, "error": error.message}
            log_action(address, request, response)
            return encode(response)
        except (UnicodeError, ValueError, RecursionError):
            response = {"id": request_id, "status": 400,
                        "error": "Некорректный JSON или данные запроса"}
            log_action(address, request, response)
            return encode(response)

    def dispatch(self, request):
        method = request.get("method")
        path = request.get("path")
        body = request.get("body", {})
        if not isinstance(path, str) or not isinstance(body, dict):
            raise RequestError(400, "Нужны строковый path и объект body")
        url = urlsplit(path)
        parts = url.path.strip("/").split("/")

        if method == "POST" and parts == ["rooms"]:
            if len(self.rooms) >= 1000:
                raise RequestError(503, "Достигнут лимит комнат")
            code = secrets.token_hex(3).upper()
            while code in self.rooms:
                code = secrets.token_hex(3).upper()
            token = secrets.token_urlsafe(32)
            self.rooms[code] = {"doctor": token, "patient": None,
                                "events": [], "created": time.monotonic(),
                                "last_frame": None}
            self.rooms[code].update(messages=[], images={}, message_seq=0)
            return 201, {"room": code, "token": token, "role": "doctor"}

        if len(parts) != 3 or parts[0] != "rooms":
            raise RequestError(404, "Неизвестный путь")
        code, action = parts[1:]
        room = self.rooms.get(code)
        if room is None:
            raise RequestError(404, "Комната не найдена или срок её жизни истёк")

        if method == "POST" and action == "join":
            if room["patient"] is not None:
                raise RequestError(409, "В комнате уже есть пациент")
            room["patient"] = secrets.token_urlsafe(32)
            return 200, {"room": code, "token": room["patient"], "role": "patient"}

        token = request.get("token")
        role = next((role for role in ("doctor", "patient")
                     if isinstance(token, str) and room[role] is not None
                     and secrets.compare_digest(token.encode(), room[role].encode())), None)
        if role is None:
            raise RequestError(401, "Неверный или отсутствующий токен")

        if action == "leave" and method == "POST":
            if role == "doctor":
                del self.rooms[code]
            else:
                room["patient"] = None
                # Новый пациент не должен получать сигналы предыдущего сеанса.
                room["events"].clear()
                room["last_frame"] = None
                room["messages"].clear()
                room["images"].clear()
                room.get("media", {}).pop("patient", None)
            return 200, {"room": code, "left": True, "closed": role == "doctor"}

        if action in ("messages", "images"):
            return self.chat(room, role, method, action, body, parse_qs(url.query))

        if action == "events" and method == "POST":
            if body.get("type") not in ("offer", "answer", "ice") or "data" not in body:
                raise RequestError(400, "Нужны type (offer, answer, ice) и data")
            if len(encode(body)) > 16000:
                raise RequestError(413, "Событие должно быть не больше 16000 байт")
            if len(room["events"]) >= 1000:
                raise RequestError(409, "Лимит событий исчерпан; создайте новую комнату")
            event = {"id": len(room["events"]) + 1, "from": role,
                     "type": body["type"], "data": body["data"]}
            room["events"].append(event)
            return 202, {"event_id": event["id"]}

        if action == "events" and method == "GET":
            try:
                after = int(parse_qs(url.query).get("after", ["0"])[0])
            except ValueError:
                raise RequestError(400, "after должен быть целым числом")
            if after < 0 or after > len(room["events"]):
                raise RequestError(400, "after выходит за пределы истории событий")
            # Ограничиваем страницу, чтобы ответ помещался в одну датаграмму.
            events = [event for event in room["events"]
                      if event["id"] > after and event["from"] != role][:3]
            return 200, {"events": events, "next_after": events[-1]["id"] if events else after}

        if action == "frames" and method == "POST":
            if role != "patient":
                raise RequestError(403, "Кадры может отправлять только пациент")
            value = body.get("jpeg_base64")
            if not isinstance(value, str):
                raise RequestError(400, "Нужно поле jpeg_base64")
            try:
                frame = base64.b64decode(value, validate=True)
            except (binascii.Error, ValueError):
                raise RequestError(400, "Некорректный Base64")
            # Проверка маркеров, не полноценное декодирование изображения.
            if not frame.startswith(b"\xff\xd8\xff") or not frame.endswith(b"\xff\xd9"):
                raise RequestError(400, "Ожидается JPEG с маркерами начала и конца")
            now = time.monotonic()
            if room["last_frame"] is not None and now - room["last_frame"] < 1:
                raise RequestError(429, "Допускается не больше одного кадра в секунду")
            result = analyze(frame)
            room["last_frame"] = now
            return 200, result

        raise RequestError(404, "Неизвестная операция")

    def chat(self, room, role, method, action, body, query):
        recipient = room["patient" if role == "doctor" else "doctor"]
        if method == "POST" and recipient is None:
            raise RequestError(409, "Собеседник ещё не вошёл в комнату")
        if method == "GET" and action == "messages":
            after = int(query.get("after", ["0"])[0])
            if after < 0:
                raise RequestError(400, "after должен быть неотрицательным")
            messages = [m for m in room["messages"]
                        if m["id"] > after and m["from"] != role][:1]
            return 200, {"messages": messages,
                         "next_after": messages[-1]["id"] if messages else after}
        if method == "POST" and action == "messages":
            text = body.get("text")
            if not isinstance(text, str) or not text.strip() or len(text.encode()) > 4000:
                raise RequestError(400, "Нужен непустой текст до 4000 байт UTF-8")
            if len(room["messages"]) >= 1000:
                raise RequestError(409, "История заполнена; создайте новую комнату")
            room["message_seq"] += 1
            message = {"id": room["message_seq"], "from": role, "type": "text", "text": text}
            room["messages"].append(message)
            return 201, {"message_id": message["id"]}
        if action != "images":
            raise RequestError(404, "Неизвестная операция")
        image_id = body.get("image_id") if method == "POST" else query.get("id", [""])[0]
        if not isinstance(image_id, str) or len(image_id) != 32 or any(c not in "0123456789abcdef" for c in image_id):
            raise RequestError(400, "Неверный идентификатор картинки")
        index = body.get("index") if method == "POST" else int(query.get("index", ["-1"])[0])
        if type(index) is not int or index < 0:
            raise RequestError(400, "Неверный номер части")
        entry = room["images"].get(image_id)
        if method == "GET":
            if entry is None or not entry["complete"]:
                raise RequestError(404, "Картинка не найдена")
            if entry["owner"] == role:
                raise RequestError(403, "Это ваша картинка")
            if index >= len(entry["chunks"]):
                raise RequestError(400, "Неверный номер части")
            return 200, {"data": base64.b64encode(entry["chunks"][index]).decode()}
        if method != "POST":
            raise RequestError(404, "Неизвестная операция")
        size, digest, extension = body.get("size"), body.get("sha256"), body.get("extension")
        if type(size) is not int or not 0 < size <= MAX_IMAGE:
            raise RequestError(413, "Максимальный размер картинки — 5 МБ")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RequestError(400, "Неверная контрольная сумма")
        if extension not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif"):
            raise RequestError(400, "Неподдерживаемое расширение картинки")
        try:
            chunk = base64.b64decode(body.get("data", ""), validate=True)
        except (ValueError, TypeError, binascii.Error):
            raise RequestError(400, "Некорректная часть картинки")
        total = (size + IMAGE_CHUNK - 1) // IMAGE_CHUNK
        expected = min(IMAGE_CHUNK, size - index * IMAGE_CHUNK)
        if index >= total or len(chunk) != expected:
            raise RequestError(400, "Некорректный размер части")
        if entry is None:
            if len(room["images"]) >= 20 or sum(e["size"] for e in room["images"].values()) + size > 20 * 1024 * 1024:
                raise RequestError(413, "Лимит картинок комнаты: 20 файлов / 20 МБ")
            entry = {"owner": role, "size": size, "sha256": digest, "extension": extension,
                     "chunks": {}, "complete": False}
            room["images"][image_id] = entry
        if (entry["owner"], entry["size"], entry["sha256"], entry["extension"]) != (role, size, digest, extension):
            raise RequestError(409, "Идентификатор занят другой картинкой")
        if index in entry["chunks"] and entry["chunks"][index] != chunk:
            raise RequestError(409, "Содержимое части отличается от уже принятого")
        entry["chunks"][index] = chunk
        if len(entry["chunks"]) == total and not entry["complete"]:
            data = b"".join(entry["chunks"][i] for i in range(total))
            if hashlib.sha256(data).hexdigest() != digest:
                del room["images"][image_id]
                raise RequestError(400, "Контрольная сумма картинки не совпала")
            if len(room["messages"]) >= 1000:
                raise RequestError(409, "История заполнена; создайте новую комнату")
            room["message_seq"] += 1
            room["messages"].append({"id": room["message_seq"], "from": role, "type": "image",
                                      "image_id": image_id, "size": size, "sha256": digest,
                                      "extension": extension, "chunks": total})
            entry["complete"] = True
        return 200, {"complete": entry["complete"]}


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        packet, sock = self.request
        if packet.startswith(PREFIX):
            for response, address in relay(self.server.application.rooms, packet, self.client_address,
                                           analysis=getattr(self.server, "analysis", None)):
                try:
                    sock.sendto(response, address)
                except OSError:
                    logger.warning("Не удалось передать медиапакет клиенту %s:%s", *address)
            return
        response = self.server.application.handle(packet, self.client_address)
        sock.sendto(response, self.client_address)


class UDPServer(socketserver.UDPServer):
    max_packet_size = 65535

    def __init__(self, address):
        self.application = Application()
        super().__init__(address, Handler)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stdout)
    parser = argparse.ArgumentParser(description="Учебный UDP-сервер ВКС")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    with UDPServer((args.host, args.port)) as server:
        print(f"UDP-сервер запущен: {args.host}:{args.port}. Остановка: Ctrl+C")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nСервер остановлен")


if __name__ == "__main__":
    main()
