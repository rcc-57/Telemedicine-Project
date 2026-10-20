"""Окно видеозвонка. Зависимости загружаются только при открытии окна."""

import io
import threading
import time
from collections import deque

from media import MediaConnection, SAMPLE_RATE, AUDIO_FRAMES, AUDIO_BYTES


def analysis_text(result):
    if not isinstance(result, dict):
        return "Анализ головы: ожидание данных"
    state = result.get("state")
    states = {"disabled": "Анализ выключен: запустите HeadAnalysisServer.py",
              "waiting": "Анализ головы: ожидание камеры пациента",
              "camera_off": "Анализ головы: камера пациента выключена",
              "stale": "Анализ головы: нет свежих кадров",
              "no_face": "Лицо не найдено — посмотрите в камеру",
              "multiple_faces": "В кадре несколько лиц — оставьте только пациента",
              "pose_unavailable": "Лицо найдено, положение головы определить не удалось",
              "calibrating": "Калибровка: смотрите прямо в камеру и не двигайте головой 2 секунды",
              "error": "Ошибка анализа — проверьте журнал сервера"}
    if state != "tracking":
        return states.get(state, "Анализ головы: ожидание данных")
    angles = result.get("angles", {})
    directions = {"center": "прямо", "left": "влево", "right": "вправо", "up": "вверх", "down": "вниз"}
    gestures = {"none": "—", "nod": "кивок", "shake": "покачивание влево-вправо",
                "turn_left": "поворот влево", "turn_right": "поворот вправо"}
    return (f"Лицо найдено | Поворот: {directions.get(result.get('turn'), '—')} "
            f"({angles.get('yaw', '—')}°) | Вверх/вниз: {directions.get(result.get('vertical'), '—')} "
            f"({angles.get('pitch', '—')}°) | Наклон к плечу: {directions.get(result.get('tilt'), '—')} "
            f"({angles.get('roll', '—')}°)\n"
            f"Движение: {'есть' if result.get('moving') else 'нет'} | "
            f"Событие: {gestures.get(result.get('gesture'), '—')} | "
            f"Кивки: {result.get('nods', 0)} | Покачивания: {result.get('shakes', 0)}")


def open_call(server, room, token, camera_index=0, input_device=None, output_device=None, role="patient"):
    try:
        import cv2
        import sounddevice as sd
        import tkinter as tk
        from tkinter import ttk
        from PIL import Image, ImageTk
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "Установите зависимости: python3 -m pip install -r requirements-client.txt. "
            f"Также нужны Tkinter и PortAudio. Подробности: {error}") from error
    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(f"Не удалось открыть графическое окно: {error}") from error
    try:
        connection = MediaConnection(server, room, token)
    except OSError:
        root.destroy()
        raise
    window = CallWindow(root, connection, cv2, sd, Image, ImageTk,
                        tk, ttk, room, camera_index, input_device, output_device, role)
    try:
        root.mainloop()
    finally:
        window.close()


class CallWindow:
    def __init__(self, root, connection, cv2, sd, Image, ImageTk,
                 tk, ttk, room, camera_index, input_device, output_device, role="patient"):
        self.root, self.connection = root, connection
        self.cv2, self.sd, self.Image, self.ImageTk = cv2, sd, Image, ImageTk
        self.camera_index, self.input_device, self.output_device = camera_index, input_device, output_device
        self.role = role
        self.closed = False
        self.camera_enabled = threading.Event()
        self.stop = threading.Event()
        self.preview_lock = threading.Lock()
        self.preview = None
        self.notices = deque(maxlen=10)
        self.input_stream = self.output_stream = None
        self.photo_local = self.photo_remote = None
        self.last_remote = None
        root.title(f"Видеозвонок — комната {room}")
        root.geometry("1040x660")
        root.minsize(960, 580)
        root.protocol("WM_DELETE_WINDOW", self.close)
        ttk.Label(root, text=f"Комната: {room}", font=("Arial", 18)).pack(pady=12)
        panels = ttk.Frame(root)
        panels.pack(fill="both", expand=True, padx=16)
        panels.columnconfigure(0, weight=1)
        panels.columnconfigure(1, weight=1)
        panels.rowconfigure(1, weight=1)
        ttk.Label(panels, text="Вы").grid(row=0, column=0, pady=5)
        ttk.Label(panels, text="Собеседник").grid(row=0, column=1, pady=5)
        self.local = tk.Label(panels, text="Камера выключена", bg="#18202b", fg="white")
        self.remote = tk.Label(panels, text="Ожидание собеседника…", bg="#18202b", fg="white")
        self.local.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.remote.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
        buttons = ttk.Frame(root)
        buttons.pack(pady=12)
        self.camera_button = ttk.Button(buttons, text="Включить камеру", command=self.toggle_camera)
        self.camera_button.pack(side="left", padx=5)
        self.microphone_button = ttk.Button(buttons, text="Включить микрофон", command=self.toggle_microphone)
        self.microphone_button.pack(side="left", padx=5)
        self.speaker_button = ttk.Button(buttons, text="Включить звук", command=self.toggle_speaker)
        self.speaker_button.pack(side="left", padx=5)
        ttk.Button(buttons, text="Калибровка головы", command=connection.calibrate).pack(side="left", padx=5)
        ttk.Button(buttons, text="Завершить звонок", command=self.close).pack(side="left", padx=5)
        self.status = tk.StringVar(value="Подключение…")
        self.notice = tk.StringVar(value="Включите камеру и микрофон кнопками. Для разговора используйте наушники.")
        ttk.Label(root, textvariable=self.status).pack(pady=3)
        self.analysis_status = tk.StringVar(value="Анализ головы: ожидание данных")
        ttk.Label(root, textvariable=self.analysis_status, wraplength=1000, justify="center").pack(pady=5)
        ttk.Label(root, text="Углы относительно калибровки. Направления — в незеркальном видео врача.").pack()
        ttk.Label(root, textvariable=self.notice, wraplength=930).pack(padx=12, pady=(3, 12))
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True, name="Webcam")
        self.camera_thread.start()
        self.toggle_speaker()
        root.after(40, self.tick)

    def toggle_camera(self):
        enabled = not self.camera_enabled.is_set()
        if enabled:
            self.camera_enabled.set()
            self.camera_button.configure(text="Выключить камеру")
        else:
            self.camera_enabled.clear()
            self.camera_button.configure(text="Включить камеру")
            with self.preview_lock:
                self.preview = None
        self.connection.controls(camera=enabled)

    def toggle_microphone(self):
        if self.input_stream is not None:
            self.connection.controls(microphone=False)
            self.close_stream("input_stream")
            self.microphone_button.configure(text="Включить микрофон")
            return
        stream = None
        try:
            stream = self.sd.RawInputStream(
                samplerate=SAMPLE_RATE, blocksize=AUDIO_FRAMES,
                channels=1, dtype="int16", device=self.input_device,
                callback=self.record_audio)
            stream.start()
            self.input_stream = stream
            self.connection.controls(microphone=True)
            self.microphone_button.configure(text="Выключить микрофон")
            self.notice.set("Микрофон включён.")
        except Exception as error:
            if stream is not None:
                stream.close()
            self.connection.controls(microphone=False)
            self.notice.set(f"Микрофон недоступен: {error}. Проверьте разрешение и устройство ввода.")

    def toggle_speaker(self):
        if self.output_stream is not None:
            self.connection.controls(speaker=False)
            self.close_stream("output_stream")
            self.speaker_button.configure(text="Включить звук")
            return
        stream = None
        try:
            stream = self.sd.RawOutputStream(
                samplerate=SAMPLE_RATE, blocksize=AUDIO_FRAMES,
                channels=1, dtype="int16", device=self.output_device,
                callback=self.play_audio)
            stream.start()
            self.output_stream = stream
            self.connection.controls(speaker=True)
            self.speaker_button.configure(text="Выключить звук")
        except Exception as error:
            if stream is not None:
                stream.close()
            self.connection.controls(speaker=False)
            self.notice.set(f"Звук недоступен: {error}. Выберите другое устройство вывода.")

    def record_audio(self, indata, frames, timing, status):
        if not self.stop.is_set() and frames == AUDIO_FRAMES:
            self.connection.capture_audio(bytes(indata))

    def play_audio(self, outdata, frames, timing, status):
        data = self.connection.playback_audio() if not self.stop.is_set() else bytes(AUDIO_BYTES)
        outdata[:] = data if len(outdata) == len(data) else bytes(len(outdata))

    def close_stream(self, attribute):
        stream = getattr(self, attribute)
        setattr(self, attribute, None)
        if stream is not None:
            try:
                stream.abort()
            finally:
                stream.close()

    def camera_loop(self):
        capture = None
        try:
            while not self.stop.is_set():
                if not self.camera_enabled.is_set():
                    if capture is not None:
                        capture.release()
                        capture = None
                    self.stop.wait(0.05)
                    continue
                if capture is None:
                    capture = self.cv2.VideoCapture(self.camera_index)
                    capture.set(self.cv2.CAP_PROP_FRAME_WIDTH, 480)
                    capture.set(self.cv2.CAP_PROP_FRAME_HEIGHT, 360)
                    if not capture.isOpened():
                        raise RuntimeError("Не удалось открыть камеру. Проверьте разрешение и --camera.")
                started = time.monotonic()
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Не удалось получить кадр с камеры.")
                if not self.camera_enabled.is_set():
                    continue
                frame = self.cv2.resize(frame, (480, 360))
                with self.preview_lock:
                    self.preview = self.cv2.cvtColor(self.cv2.flip(frame, 1), self.cv2.COLOR_BGR2RGB)
                ok, jpeg = self.cv2.imencode(".jpg", frame, [self.cv2.IMWRITE_JPEG_QUALITY, 50])
                if ok:
                    try:
                        self.connection.send_video(jpeg.tobytes())
                    except OSError:
                        pass  # Состояние сети показывает медиапоток.
                self.stop.wait(max(0, 1 / 12 - (time.monotonic() - started)))
        except Exception as error:
            self.camera_enabled.clear()
            self.connection.controls(camera=False)
            self.notices.append(f"Камера: {error}")
        finally:
            if capture is not None:
                capture.release()

    def show_image(self, label, image, attribute):
        with self.connection.lock:
            result = self.connection.analysis
        is_patient_view = (attribute == "photo_local" and self.role == "patient") or (attribute == "photo_remote" and self.role == "doctor")
        if is_patient_view and isinstance(result, dict) and result.get("state") in ("tracking", "calibrating"):
            bbox = result.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4 and all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in bbox):
                from PIL import ImageDraw
                x1, y1, x2, y2 = bbox
                if attribute == "photo_local":
                    x1, x2 = 1 - x2, 1 - x1
                if x2 > x1 and y2 > y1:
                    ImageDraw.Draw(image).rectangle((x1 * image.width, y1 * image.height,
                                                    x2 * image.width, y2 * image.height), outline="#60e080", width=3)
        image.thumbnail((max(1, label.winfo_width()), max(1, label.winfo_height())))
        photo = self.ImageTk.PhotoImage(image)
        setattr(self, attribute, photo)
        label.configure(image=photo, text="")

    def tick(self):
        if self.closed:
            return
        while self.notices:
            self.notice.set(self.notices.popleft())
            self.camera_button.configure(text="Включить камеру")
        if self.connection.fatal:
            self.camera_enabled.clear()
            self.close_stream("input_stream")
            self.camera_button.configure(state="disabled")
            self.microphone_button.configure(state="disabled")
        # После ошибки камеры разрешаем повторную попытку теми же кнопками.
        if not self.camera_thread.is_alive() and not self.stop.is_set():
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True, name="Webcam")
            self.camera_thread.start()
        with self.preview_lock:
            preview = self.preview
        if self.camera_enabled.is_set() and preview is not None:
            self.show_image(self.local, self.Image.fromarray(preview), "photo_local")
        else:
            self.local.configure(image="", text="Камера выключена")
            self.photo_local = None
        with self.connection.lock:
            jpeg = self.connection.video
            video_time = self.connection.video_time
            status = self.connection.status
            peer_camera = self.connection.peer_camera
            peer_microphone = self.connection.peer_microphone
            analysis = self.connection.analysis
        self.analysis_status.set(analysis_text(analysis))
        self.status.set(f"{status} | Камера собеседника: {'вкл' if peer_camera else 'выкл'} | "
                        f"Микрофон: {'вкл' if peer_microphone else 'выкл'}")
        if jpeg is not None and time.monotonic() - video_time < 2:
            if jpeg is not self.last_remote:
                try:
                    with self.Image.open(io.BytesIO(jpeg)) as source:
                        if source.width > 1920 or source.height > 1080:
                            raise ValueError("Слишком большой кадр")
                        image = source.convert("RGB")
                    self.show_image(self.remote, image, "photo_remote")
                except (OSError, ValueError):
                    pass
                self.last_remote = jpeg
        else:
            self.remote.configure(image="", text="Нет видео" if peer_camera else "Камера собеседника выключена")
            self.photo_remote = self.last_remote = None
        self.root.after(40, self.tick)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop.set()
        self.camera_enabled.clear()
        self.connection.controls(camera=False, microphone=False, speaker=False)
        try:
            for attribute in ("input_stream", "output_stream"):
                try:
                    self.close_stream(attribute)
                except Exception:
                    pass
            self.camera_thread.join(timeout=2)
            self.connection.close()
        finally:
            self.root.destroy()
