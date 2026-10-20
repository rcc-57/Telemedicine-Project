import base64
import json
import socket
import struct
import tempfile
from pathlib import Path
import threading
import unittest
import uuid
from collections import deque
from unittest.mock import patch

from Lab_Work_two import Application, UDPServer
from Client import Client, ServerError


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.server = UDPServer(("127.0.0.1", 0))
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01})
        self.thread.start()
        self.client = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.client.settimeout(2)
        self.peer = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.peer.settimeout(2)
        for client in (self.client, self.peer):
            client.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65535)

    def tearDown(self):
        self.client.close()
        self.peer.close()
        self.server.shutdown()
        self.thread.join()
        self.server.server_close()

    def call(self, method, path, token=None, body=None, request_id=None, peer=False):
        packet = json.dumps({"id": request_id or uuid.uuid4().hex,
                             "method": method, "path": path,
                             "token": token, "body": body or {}}).encode()
        client = self.peer if peer else self.client
        client.sendto(packet, self.server.server_address)
        return json.loads(client.recv(65535))

    def room(self):
        doctor = self.call("POST", "/rooms")["body"]
        path = "/rooms/" + doctor["room"]
        patient = self.call("POST", path + "/join", peer=True)["body"]
        return path, doctor["token"], patient["token"]

    def test_room_and_retries(self):
        first = self.call("POST", "/rooms", request_id="same")
        self.assertEqual(first["status"], 201)
        self.assertEqual(first, self.call("POST", "/rooms", request_id="same"))
        self.assertEqual(len(self.server.application.rooms), 1)
        self.assertEqual(self.call("GET", "/rooms", request_id="same")["status"], 409)
        path = "/rooms/" + first["body"]["room"]
        self.assertEqual(self.call("POST", path + "/join")["status"], 200)
        self.assertEqual(self.call("POST", path + "/join")["status"], 409)

    def test_signaling_and_pagination(self):
        path, doctor, patient = self.room()
        for i in range(5):
            response = self.call("POST", path + "/events", doctor,
                                 {"type": "offer", "data": {"sdp": "x" * 15000}},
                                 request_id=str(i))
            self.assertEqual(response["status"], 202)
        page = self.call("GET", path + "/events?after=0", patient, peer=True)["body"]
        self.assertEqual(len(page["events"]), 3)
        rest = self.call("GET", path + "/events?after=3", patient, peer=True)["body"]
        self.assertEqual(len(rest["events"]), 2)
        self.assertEqual(self.call("GET", path + "/events", doctor)["body"]["events"], [])
        self.assertEqual(self.call("GET", path + "/events", "bad")["status"], 401)
        self.assertEqual(self.call("GET", path + "/events?after=-1", patient)["status"], 400)
        self.call("POST", path + "/events", patient,
                  {"type": "answer", "data": "answer"}, peer=True)
        self.assertEqual(self.call("GET", path + "/events", doctor)["body"]["events"][0]["type"], "answer")

    def test_frame_access_validation_and_rate(self):
        path, doctor, patient = self.room()
        # Минимальная последовательность маркеров для проверки транспорта.
        body = {"jpeg_base64": base64.b64encode(b"\xff\xd8\xff\xd9").decode()}
        self.assertEqual(self.call("POST", path + "/frames", doctor, body)["status"], 403)
        self.assertEqual(self.call("POST", path + "/frames", patient,
                                   {"jpeg_base64": "bad"})["status"], 400)
        self.assertEqual(self.call("POST", path + "/frames", patient, body)["status"], 200)
        self.assertEqual(self.call("POST", path + "/frames", patient, body)["status"], 429)

    def test_bad_packets_do_not_stop_server(self):
        for packet in (b"not json", b"[]", b"\xff", b"x" * 60001):
            self.client.sendto(packet, self.server.server_address)
            response = json.loads(self.client.recv(65535))
            self.assertIn(response["status"], (400, 413))
        self.assertEqual(self.call("POST", "/rooms")["status"], 201)

    def test_client_join_leave_and_rejoin(self):
        doctor = Client(*self.server.server_address)
        patient = Client(*self.server.server_address)
        other = Client(*self.server.server_address)
        try:
            doctor.enter()
            code = doctor.room
            patient.enter(code.lower())
            old_token = patient.token
            with self.assertRaises(ServerError) as error:
                other.enter(code)
            self.assertEqual(error.exception.status, 409)
            with self.assertRaises(ValueError):
                patient.enter(code)
            self.call("POST", f"/rooms/{code}/events", doctor.token,
                      {"type": "offer", "data": "old session"})
            patient.leave()
            self.assertIsNone(patient.room)
            self.assertEqual(self.call("GET", f"/rooms/{code}/events", old_token)["status"], 401)
            patient.enter(code)
            self.assertNotEqual(patient.token, old_token)
            self.assertEqual(self.call("GET", f"/rooms/{code}/events", patient.token)["body"]["events"], [])
            patient.leave()
            other.enter(code)
            doctor.leave()
            self.assertNotIn(code, self.server.application.rooms)
            other.leave()  # Закрытая врачом комната также очищает локальное состояние.
            self.assertIsNone(other.room)
        finally:
            doctor.close()
            patient.close()
            other.close()

    def test_chat_and_image_round_trip(self):
        doctor = Client(*self.server.server_address)
        patient = Client(*self.server.server_address)
        try:
            doctor.enter()
            with self.assertRaises(ServerError):
                doctor.send_message("Пока никого нет")
            patient.enter(doctor.room)
            doctor.send_message("Здравствуйте! Как самочувствие?")
            patient.send_message("Добрый день!")
            self.assertEqual(list(patient.receive_messages())[0]["text"], "Здравствуйте! Как самочувствие?")
            self.assertEqual(list(doctor.receive_messages())[0]["text"], "Добрый день!")
            self.assertEqual(list(patient.receive_messages()), [])
            with tempfile.TemporaryDirectory() as folder:
                # Настоящий BMP 128×128, существенно больше одной датаграммы.
                pixels = b"\x00\x80\xff" * (128 * 128)
                data = (b"BM" + struct.pack("<IHHI", 54 + len(pixels), 0, 0, 54)
                        + struct.pack("<IiiHHIIiiII", 40, 128, 128, 1, 24, 0,
                                      len(pixels), 0, 0, 0, 0) + pixels)
                source = Path(folder) / "sample.bmp"
                source.write_bytes(data)
                patient.send_image(str(source))
                messages = list(doctor.receive_messages(Path(folder) / "doctor"))
                self.assertEqual(len(messages), 1)
                self.assertEqual(Path(messages[0]["saved_path"]).read_bytes(), data)
                self.assertEqual(list(doctor.receive_messages()), [])
                doctor.send_image(str(source))
                messages = list(patient.receive_messages(Path(folder) / "patient"))
                self.assertEqual(Path(messages[0]["saved_path"]).read_bytes(), data)
            old_cursor = doctor.after
            patient.leave()
            patient.enter(doctor.room)
            patient.send_message("Новый сеанс")
            message = list(doctor.receive_messages())[0]
            self.assertGreater(message["id"], old_cursor)
            self.assertEqual(message["text"], "Новый сеанс")
        finally:
            doctor.close()
            patient.close()

    def test_image_parts_validation_and_duplicate(self):
        import hashlib
        path, doctor, patient = self.room()
        data = b"test image transport" * 100
        body = {"image_id": uuid.uuid4().hex, "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(), "extension": ".png"}
        for index, start in enumerate(range(0, len(data), 768)):
            body.update(index=index, data=base64.b64encode(data[start:start + 768]).decode())
            self.assertEqual(self.call("POST", path + "/images", doctor, body)["status"], 200)
            # Другой request id: сервер также подавляет повтор по номеру части.
            self.assertEqual(self.call("POST", path + "/images", doctor, body)["status"], 200)
        messages = self.call("GET", path + "/messages", patient)["body"]["messages"]
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(self.server.application.rooms[path.split('/')[2]]["messages"]), 1)
        query = path + "/images?id=" + body["image_id"] + "&index=0"
        self.assertEqual(self.call("GET", query, "bad")["status"], 401)
        self.assertEqual(self.call("GET", query, doctor)["status"], 403)
        self.assertEqual(self.call("GET", query, patient)["status"], 200)
        body.update(index=0, data=base64.b64encode(b"x" * 768).decode())
        self.assertEqual(self.call("POST", path + "/images", doctor, body)["status"], 409)
        body.update(image_id=uuid.uuid4().hex, size=1, data="eA==", sha256="0" * 64)
        self.assertEqual(self.call("POST", path + "/images", doctor, body)["status"], 400)


class ChatLogicTests(unittest.TestCase):
    def test_lost_image_ack_and_resumed_download(self):
        """Имитируем потерю ответов без привязки к сетевым портам."""
        application = Application()

        class Transport:
            next_port = 51000

            def __init__(self, *args):
                self.address = ("127.0.0.1", Transport.next_port)
                Transport.next_port += 1
                self.queue = deque()
                self.drop_upload_ack = False
                self.drop_download = False

            def setsockopt(self, *args):
                pass

            def connect(self, *args):
                pass

            def settimeout(self, *args):
                pass

            def close(self):
                pass

            def send(self, packet):
                request = json.loads(packet)
                response = application.handle(packet, self.address)
                if request["method"] == "POST" and request["path"].endswith("/images") and self.drop_upload_ack:
                    self.drop_upload_ack = False
                    return
                if request["method"] == "GET" and "/images?" in request["path"] and self.drop_download:
                    return
                self.queue.append(response)

            def recv(self, size):
                if not self.queue:
                    raise socket.timeout()
                return self.queue.popleft()

        with patch("Client.socket.socket", Transport), tempfile.TemporaryDirectory() as folder:
            doctor, patient = Client(), Client()
            doctor.enter()
            patient.enter(doctor.room)
            source = Path(folder) / "test.png"
            # Проверяем передачу байтов; декодирование картинок не входит в протокол.
            data = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 10
            source.write_bytes(data)
            patient.sock.drop_upload_ack = True
            patient.send_image(str(source))
            self.assertEqual(len(application.rooms[doctor.room]["messages"]), 1)
            doctor.sock.drop_download = True
            with self.assertRaises(TimeoutError):
                list(doctor.receive_messages(Path(folder) / "inbox"))
            self.assertEqual(doctor.after, 0)
            doctor.sock.drop_download = False
            received = list(doctor.receive_messages(Path(folder) / "inbox"))
            self.assertEqual(Path(received[0]["saved_path"]).read_bytes(), data)
            self.assertEqual(list(doctor.receive_messages()), [])
            doctor.close()
            patient.close()


if __name__ == "__main__":
    unittest.main()
