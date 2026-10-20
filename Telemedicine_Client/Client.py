"""Консольный UDP-клиент: комнаты, сообщения и картинки."""

import argparse
import base64
import hashlib
import json
import socket
import time
import uuid
from pathlib import Path


IMAGE_CHUNK = 768
MAX_IMAGE = 5 * 1024 * 1024


class ServerError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Client:
    def __init__(self, host="127.0.0.1", port=5000):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)
        try:
            self.sock.connect((host, port))
        except OSError:
            self.sock.close()
            raise
        self.room = None
        self.token = None
        self.role = None
        self.pending = None
        self.after = 0
        self.upload = None

    def request(self, path, method="POST", body=None):
        # Сохраняем id при тайм-ауте: повтор команды не создаст второй сеанс.
        signature = json.dumps([method, path, body], sort_keys=True)
        if self.pending is None:
            request_id = uuid.uuid4().hex
            packet = json.dumps({"id": request_id, "method": method,
                                 "path": path, "token": self.token,
                                 "body": body or {}}, ensure_ascii=False).encode()
            if method == "POST":
                self.pending = (signature, request_id, packet)
        else:
            pending_signature, request_id, packet = self.pending
            if pending_signature != signature:
                raise ValueError("Сначала повторите предыдущую команду с теми же данными: её результат неизвестен.")
        for _ in range(3):
            self.sock.send(packet)
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                self.sock.settimeout(max(0.001, deadline - time.monotonic()))
                try:
                    response = json.loads(self.sock.recv(65535))
                except socket.timeout:
                    break
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(response, dict) or response.get("id") != request_id:
                    continue
                if not isinstance(response.get("status"), int):
                    continue
                self.pending = None
                if response["status"] >= 400:
                    raise ServerError(response["status"], response.get("error", "Ошибка сервера"))
                return response["body"]
        raise TimeoutError("Нет ответа сервера. Проверьте адрес и повторите команду.")

    def enter(self, code=None):
        if self.room is not None:
            raise ValueError("Сначала выйдите из текущей комнаты.")
        if code is None:
            path = "/rooms"
        else:
            code = code.strip().upper()
            if len(code) != 6 or any(char not in "0123456789ABCDEF" for char in code):
                raise ValueError("Код комнаты — 6 символов: цифры и буквы A–F.")
            path = f"/rooms/{code}/join"
        result = self.request(path)
        self.room, self.token, self.role = result["room"], result["token"], result["role"]
        self.after = 0
        self.upload = None

    def leave(self):
        if self.room is None:
            raise ValueError("Вы ещё не вошли в комнату.")
        try:
            self.request(f"/rooms/{self.room}/leave")
        except ServerError as error:
            # Комната могла закрыться или истечь; действующего участия уже нет.
            if error.status not in (401, 404):
                raise
        self.room = self.token = self.role = None
        self.after = 0
        self.upload = None

    def room_path(self):
        if self.room is None:
            raise ValueError("Сначала войдите в комнату.")
        return f"/rooms/{self.room}"

    def send_message(self, text):
        path = self.room_path()
        if not text.strip() or len(text.encode("utf-8")) > 4000:
            raise ValueError("Введите непустое сообщение до 4000 байт UTF-8.")
        return self.request(path + "/messages", body={"text": text})

    def send_image(self, filename):
        room_path = self.room_path()
        path = Path(filename.strip().strip('\"').strip("'")).expanduser().resolve()
        if self.upload is not None and self.upload["path"] != path:
            raise ValueError("Сначала повторите отправку предыдущей картинки.")
        if self.upload is None:
            extension = path.suffix.lower()
            if extension not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif"):
                raise ValueError("Выберите JPG, PNG, GIF, WebP, BMP или HEIC/HEIF.")
            with path.open("rb") as source:
                data = source.read(MAX_IMAGE + 1)
            if not 0 < len(data) <= MAX_IMAGE:
                raise ValueError("Размер картинки должен быть от 1 байта до 5 МБ.")
            self.upload = {"path": path, "data": data, "index": 0,
                           "image_id": uuid.uuid4().hex, "extension": extension,
                           "sha256": hashlib.sha256(data).hexdigest()}
        upload = self.upload
        data = upload["data"]
        total = (len(data) + IMAGE_CHUNK - 1) // IMAGE_CHUNK
        while upload["index"] < total:
            index = upload["index"]
            chunk = data[index * IMAGE_CHUNK:(index + 1) * IMAGE_CHUNK]
            result = self.request(room_path + "/images", body={
                "image_id": upload["image_id"], "index": index, "size": len(data),
                "extension": upload["extension"], "sha256": upload["sha256"],
                "data": base64.b64encode(chunk).decode()})
            upload["index"] += 1
        self.upload = None
        if not result["complete"]:
            raise ValueError("Сеанс изменился во время передачи. Отправьте картинку заново.")

    def download_image(self, message, directory):
        image_id = message["image_id"]
        extension = message["extension"]
        if (not isinstance(image_id, str) or len(image_id) != 32
                or any(c not in "0123456789abcdef" for c in image_id)
                or extension not in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".heic", ".heif")):
            raise ValueError("Некорректные сведения о картинке.")
        size, total = message["size"], message["chunks"]
        if type(size) is not int or not 0 < size <= MAX_IMAGE or total != (size + IMAGE_CHUNK - 1) // IMAGE_CHUNK:
            raise ValueError("Некорректный размер картинки.")
        data = bytearray()
        for index in range(total):
            result = self.request(self.room_path() + f"/images?id={image_id}&index={index}", method="GET")
            chunk = base64.b64decode(result["data"], validate=True)
            if len(chunk) != min(IMAGE_CHUNK, size - index * IMAGE_CHUNK):
                raise ValueError("Некорректный размер части картинки.")
            data.extend(chunk)
        if hashlib.sha256(data).hexdigest() != message["sha256"]:
            raise ValueError("Контрольная сумма картинки не совпала. Повторите получение.")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Имя создаём сами, без путей отправителя. Существующие файлы не перезаписываем.
        path = directory / f"{image_id}_{uuid.uuid4().hex[:8]}{extension}"
        with path.open("xb") as output:
            output.write(data)
        return path.resolve()

    def receive_messages(self, directory="received_images"):
        """Выдаёт входящие по одному, сохраняя курсор после успешной обработки."""
        path = self.room_path()
        while True:
            result = self.request(path + f"/messages?after={self.after}", method="GET")
            if not result["messages"]:
                return
            for message in result["messages"]:
                if message["type"] == "image":
                    message["saved_path"] = str(self.download_image(message, directory))
                self.after = message["id"]
                yield message

    def close(self):
        self.sock.close()


def main():
    parser = argparse.ArgumentParser(description="Клиент UDP-комнат")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--downloads", default="received_images", help="Папка для полученных картинок")
    parser.add_argument("--camera", type=int, default=0, help="Номер веб-камеры")
    parser.add_argument("--input-device", type=int, default=None, help="Номер микрофона sounddevice")
    parser.add_argument("--output-device", type=int, default=None, help="Номер динамиков sounddevice")
    args = parser.parse_args()
    try:
        client = Client(args.host, args.port)
    except OSError as error:
        print(f"Не удалось открыть соединение: {error}")
        return
    print("1 — создать комнату (врач)\n2 — войти по коду (пациент)\n"
          "3 — выйти из комнаты\n4 — написать сообщение\n5 — отправить картинку\n"
          "6 — проверить входящие\n7 — открыть видеозвонок\n0 — закрыть клиент")
    try:
        while True:
            try:
                command = input("\nКоманда: ").strip()
                if command == "1":
                    client.enter()
                    print(f"Вы вошли как врач. Код для пациента: {client.room}")
                elif command == "2":
                    client.enter(input("Код комнаты: "))
                    print(f"Вы вошли в комнату {client.room} как пациент.")
                elif command == "3":
                    client.leave()
                    print("Вы вышли из комнаты.")
                elif command == "4":
                    client.send_message(input("Текст: "))
                    print("Сообщение принято сервером. Собеседник может получить его командой 6.")
                elif command == "5":
                    filename = input("Путь к картинке: ")
                    print("Отправка картинки…")
                    client.send_image(filename)
                    print("Картинка принята сервером. Собеседник может получить её командой 6.")
                elif command == "6":
                    count = 0
                    for message in client.receive_messages(args.downloads):
                        sender = "Врач" if message["from"] == "doctor" else "Пациент"
                        if message["type"] == "text":
                            text = "".join(c if c.isprintable() or c == "\n" else "?" for c in message["text"])
                            print(f"{sender}: {text}")
                        else:
                            print(f"{sender} отправил картинку. Сохранена: {message['saved_path']}")
                        count += 1
                    if not count:
                        print("Новых сообщений нет.")
                elif command == "7":
                    client.room_path()
                    try:
                        from VideoCall import open_call
                        open_call(client.sock.getpeername(), client.room, client.token,
                                  camera_index=args.camera,
                                  input_device=args.input_device, output_device=args.output_device,
                                  role=client.role)
                    except (ImportError, RuntimeError) as error:
                        print(f"Не удалось открыть звонок: {error}")
                elif command == "0":
                    break
                else:
                    print("Введите команду от 0 до 7.")
            except (OSError, ValueError, ServerError) as error:
                print(f"Ошибка: {error}")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        try:
            if client.room is not None:
                client.leave()
                print("Вы вышли из комнаты.")
            elif client.pending is not None:
                print("Последний вход не подтверждён. Сервер мог сохранить участие до истечения комнаты.")
        except (OSError, ValueError, ServerError, KeyboardInterrupt):
            print("Не удалось подтвердить выход на сервере.")
        finally:
            client.close()
    print("Клиент закрыт.")


if __name__ == "__main__":
    main()
