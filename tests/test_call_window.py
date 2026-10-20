"""Проверка настоящего окна с искусственными камерой и аудиоустройствами.

Запуск вручную: RUN_GUI_TESTS=1 .venv/bin/python -m unittest -v test_call_window.py
"""
import os
import threading
import time
import unittest
from unittest.mock import Mock


@unittest.skipUnless(os.environ.get("RUN_GUI_TESTS") == "1", "Требуется графический сеанс")
class WindowTests(unittest.TestCase):
    def test_preview_and_device_buttons(self):
        import cv2
        import numpy as np
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
        from VideoCall import CallWindow
        from media import AUDIO_BYTES

        class Stream:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
                self.closed = False

            def start(self):
                pass

            def abort(self):
                pass

            def close(self):
                self.closed = True

        class Camera:
            def __init__(self, index):
                self.closed = False

            def set(self, *args):
                pass

            def isOpened(self):
                return True

            def read(self):
                return True, np.full((360, 480, 3), (80, 180, 40), dtype=np.uint8)

            def release(self):
                self.closed = True

        connection = Mock()
        connection.lock = threading.RLock()
        connection.fatal = False
        connection.video = cv2.imencode(".jpg", np.full((360, 480, 3), (200, 80, 20), dtype=np.uint8))[1].tobytes()
        connection.video_time = time.monotonic()
        connection.peer_camera = connection.peer_microphone = True
        connection.status = "Тестовый собеседник"
        connection.analysis = {"state": "tracking", "faces": 1, "bbox": [0.2, 0.15, 0.8, 0.85],
                               "angles": {"pitch": 0, "yaw": 0, "roll": 0},
                               "turn": "center", "vertical": "center", "tilt": "center"}
        connection.playback_audio.return_value = bytes(AUDIO_BYTES)
        fake_cv = Mock(wraps=cv2)
        for name in ("CAP_PROP_FRAME_WIDTH", "CAP_PROP_FRAME_HEIGHT", "COLOR_BGR2RGB", "IMWRITE_JPEG_QUALITY"):
            setattr(fake_cv, name, getattr(cv2, name))
        camera = Camera(0)
        fake_cv.VideoCapture = Mock(return_value=camera)
        audio = Mock(RawInputStream=Stream, RawOutputStream=Stream)
        root = tk.Tk()
        window = CallWindow(root, connection, fake_cv, audio, Image, ImageTk,
                            tk, ttk, "TEST01", 0, None, None)
        try:
            window.camera_button.invoke()
            window.microphone_button.invoke()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                connection.video_time = time.monotonic()
                root.update()
                time.sleep(0.01)
            connection.video_time = time.monotonic()
            window.tick()
            self.assertTrue(window.local.cget("image"))
            self.assertTrue(window.remote.cget("image"),
                            (window.remote.cget("text"), window.last_remote is None, window.status.get()))
            self.assertTrue(connection.send_video.called)
            self.assertIn("Лицо найдено", window.analysis_status.get())
            mic = window.input_stream
            window.microphone_button.invoke()
            self.assertTrue(mic.closed)
            self.assertIsNone(window.input_stream)
            connection.controls.assert_called_with(microphone=False)
            window.camera_button.invoke()
            self.assertFalse(window.camera_enabled.is_set())
            connection.controls.assert_called_with(camera=False)
            output = window.output_stream
            window.speaker_button.invoke()
            self.assertTrue(output.closed)
            connection.controls.assert_called_with(speaker=False)
        finally:
            window.close()
        self.assertTrue(camera.closed)
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
