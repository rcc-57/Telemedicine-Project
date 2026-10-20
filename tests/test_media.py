import base64
import json
import socket
import threading
import time
import unittest

from Lab_Work_two import Application, UDPServer, encode
from media import (AUDIO_BYTES, CHUNK, MediaConnection, VideoAssembler,
                   pack, relay, unpack)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.application = Application()
        doctor = self.application.dispatch({"method": "POST", "path": "/rooms"})[1]
        self.code = doctor["room"]
        patient = self.application.dispatch({"method": "POST", "path": f"/rooms/{self.code}/join"})[1]
        self.identities = [dict(room=self.code, token=doctor["token"], sid="a" * 32),
                           dict(room=self.code, token=patient["token"], sid="b" * 32)]
        self.addresses = [("127.0.0.1", 10001), ("127.0.0.1", 10002)]
        self.now = time.monotonic()
        for index in (0, 1):
            self.send(index, kind="hello", camera=True, microphone=True)

    def send(self, participant, **message):
        return relay(self.application.rooms, pack({**self.identities[participant], **message}),
                     self.addresses[participant], self.now)

    def test_relay_auth_mute_leave_and_expiry(self):
        data = base64.b64encode(bytes(AUDIO_BYTES)).decode()
        response = self.send(0, kind="audio", seq=1, data=data)
        self.assertEqual(response[0][1], self.addresses[1])
        self.assertNotIn("token", unpack(response[0][0]))
        self.send(0, kind="hello", camera=True, microphone=False)
        self.assertEqual(self.send(0, kind="audio", seq=2, data=data), [])
        self.identities[0]["token"] = "invalid"
        self.assertEqual(unpack(self.send(0, kind="audio", seq=3, data=data)[0][0])["kind"], "error")
        self.assertEqual(self.send(1, kind="audio", seq=4, data="bad"), [])
        self.send(1, kind="bye")
        self.assertNotIn("patient", self.application.rooms[self.code]["media"])
        self.now += 3601
        self.assertEqual(unpack(self.send(1, kind="hello", camera=True, microphone=True)[0][0])["kind"], "error")

    def test_assembler_reordering_duplicates_and_missing_parts(self):
        data = bytes(range(256)) * 7
        parts = [dict(kind="video", seq=1, index=i, total=3,
                      data=base64.b64encode(data[i * CHUNK:(i + 1) * CHUNK]).decode())
                 for i in range(3)]
        assembler = VideoAssembler()
        self.assertIsNone(assembler.add(parts[2], now=0))
        self.assertIsNone(assembler.add(parts[2], now=0))
        self.assertIsNone(assembler.add(parts[0], now=0))
        self.assertEqual(assembler.add(parts[1], now=0), data)
        self.assertIsNone(assembler.add(parts[0], now=0))
        parts[0]["seq"] = 2
        assembler.add(parts[0], now=0)
        parts[0]["seq"] = 3
        assembler.add(parts[0], now=1)
        self.assertNotIn(2, assembler.frames)

    def test_disabled_camera_and_wrong_address(self):
        part = dict(kind="video", seq=1, index=0, total=1, data="eA==")
        self.assertTrue(self.send(0, **part))
        self.send(0, kind="hello", camera=False, microphone=True)
        self.assertEqual(self.send(0, **part), [])
        self.addresses[1] = ("127.0.0.1", 10003)
        self.assertEqual(self.send(1, **part), [])
        self.application.dispatch({"method": "POST", "path": f"/rooms/{self.code}/leave",
                                   "token": self.identities[1]["token"]})
        self.assertNotIn("patient", self.application.rooms[self.code]["media"])


class NetworkMediaTests(unittest.TestCase):
    def test_bidirectional_media_and_control_socket(self):
        server = UDPServer(("127.0.0.1", 0))
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        connections = []
        try:
            # Создание/вход — по тому же протоколу, что и консольный клиент.
            from Client import Client
            doctor = Client(*server.server_address)
            patient = Client(*server.server_address)
            try:
                doctor.enter()
                patient.enter(doctor.room)
                for client in (doctor, patient):
                    connections.append(MediaConnection(server.server_address, client.room, client.token))
                first, second = connections
                first.controls(camera=True, microphone=True)
                second.controls(camera=True, microphone=True)
                deadline = time.monotonic() + 3
                while not (first.peer_camera and second.peer_camera) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(first.peer_camera and second.peer_camera)
                first.send_video(b"jpeg A" * 400)
                second.send_video(b"jpeg B" * 400)
                first.capture_audio(b"\x01\x02" * (AUDIO_BYTES // 2))
                second.capture_audio(b"\x03\x04" * (AUDIO_BYTES // 2))
                deadline = time.monotonic() + 2
                while (first.video is None or second.video is None or not first.in_audio or not second.in_audio) and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(first.video, b"jpeg B" * 400)
                self.assertEqual(second.video, b"jpeg A" * 400)
                self.assertEqual(first.playback_audio(), b"\x03\x04" * (AUDIO_BYTES // 2))
                self.assertEqual(second.playback_audio(), b"\x01\x02" * (AUDIO_BYTES // 2))
                first.controls(microphone=False, camera=False)
                first.capture_audio(b"\x05\x06" * (AUDIO_BYTES // 2))
                self.assertEqual(len(first.out_audio), 0)
                deadline = time.monotonic() + 2
                while second.peer_camera and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertFalse(second.peer_camera)
                self.assertIsNone(second.video)
                second.controls(speaker=False)
                self.assertEqual(second.playback_audio(), bytes(AUDIO_BYTES))
                doctor.send_message("Чат работает во время медиапередачи")
                self.assertEqual(list(patient.receive_messages())[0]["text"], "Чат работает во время медиапередачи")
                doctor.leave()
                deadline = time.monotonic() + 2
                while not second.fatal and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(second.fatal)
            finally:
                doctor.close()
                patient.close()
        finally:
            for connection in connections:
                connection.close()
            server.shutdown()
            thread.join()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
