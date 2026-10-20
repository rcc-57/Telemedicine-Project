"""Обнаружение лица, относительная поза головы и простые движения."""

import io
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path

from media import VideoAssembler, PEER_TIMEOUT


def difference(angle, reference):
    return (angle - reference + 180) % 360 - 180


class MotionTracker:
    """Пороговые эвристики движений; не определяет намерения человека."""

    def __init__(self):
        self.baseline = None
        self.samples = deque(maxlen=10)
        self.previous = None
        self.last_time = None
        self.pitch_excursion = None
        self.yaw_sides = set()
        self.yaw_started = None
        self.last_turn = "center"
        self.gesture = "none"
        self.gesture_until = 0
        self.nods = self.shakes = 0

    def reset_motion(self):
        self.previous = None
        self.pitch_excursion = None
        self.yaw_sides.clear()
        self.yaw_started = None
        self.last_turn = "center"
        self.gesture = "none"

    def update(self, face, now):
        result = dict(face)
        if self.last_time is not None and now - self.last_time > 1:
            self.baseline = None
            self.samples.clear()
            self.reset_motion()
        elapsed = now - self.last_time if self.last_time is not None else 0.125
        self.last_time = now
        if result["state"] != "face":
            self.baseline = None
            self.samples.clear()
            self.reset_motion()
            return result
        angles = result.pop("raw_angles")  # pitch, yaw, roll
        if self.baseline is None:
            self.samples.append(angles)
            stable = all(max(s[i] for s in self.samples) - min(s[i] for s in self.samples) < 5
                         for i in range(3))
            facing = abs(angles[0]) < 30 and abs(angles[1]) < 25 and abs(angles[2]) < 25
            if not stable or not facing:
                self.samples.clear()
            if len(self.samples) < 10:
                result.update(state="calibrating", progress=len(self.samples))
                return result
            self.baseline = tuple(sum(s[i] for s in self.samples) / len(self.samples) for i in range(3))
        current = tuple(difference(a, b) for a, b in zip(angles, self.baseline))
        if self.previous is not None:
            current = tuple(old + 0.65 * difference(new, old) for new, old in zip(current, self.previous))
        speed = (max(abs(difference(a, b)) for a, b in zip(current, self.previous)) / max(elapsed, 0.01)
                 if self.previous is not None else 0)
        self.previous = current
        pitch, yaw, roll = current
        if self.pitch_excursion is not None and now - self.pitch_excursion > 2.5:
            self.pitch_excursion = None
        if abs(pitch) >= 12 and self.pitch_excursion is None:
            self.pitch_excursion = now
        if abs(pitch) < 4 and self.pitch_excursion is not None:
            if 0.15 <= now - self.pitch_excursion <= 2.5:
                self.nods += 1
                self.gesture, self.gesture_until = "nod", now + 1.5
            self.pitch_excursion = None
        if self.yaw_started is not None and now - self.yaw_started > 3:
            self.yaw_sides.clear()
            self.yaw_started = None
        turn = "right" if yaw > 15 else "left" if yaw < -15 else "center"
        if turn != "center":
            if self.yaw_started is None:
                self.yaw_started = now
            self.yaw_sides.add(turn)
            if turn != self.last_turn:
                self.gesture, self.gesture_until = "turn_" + turn, now + 1.5
        elif abs(yaw) < 6 and len(self.yaw_sides) == 2:
            self.shakes += 1
            self.gesture, self.gesture_until = "shake", now + 1.5
            self.yaw_sides.clear()
            self.yaw_started = None
        self.last_turn = turn
        result.update(state="tracking", angles={"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)},
                      turn=turn, vertical="down" if pitch > 12 else "up" if pitch < -12 else "center",
                      tilt="left" if roll > 12 else "right" if roll < -12 else "center",
                      moving=speed > 8, gesture=self.gesture if now < self.gesture_until else "none",
                      nods=self.nods, shakes=self.shakes)
        return result


class FaceEngine:
    def __init__(self, model_path):
        import mediapipe as mp
        import numpy as np
        from PIL import Image
        self.mp, self.np, self.Image = mp, np, Image
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=2, min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
            output_facial_transformation_matrixes=True)
        # IMAGE не переносит внутреннее отслеживание между разными комнатами.
        self.detector = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    def detect(self, jpeg):
        with self.Image.open(io.BytesIO(jpeg)) as source:
            if source.format != "JPEG" or source.width > 1920 or source.height > 1080:
                raise ValueError("Ожидается JPEG до 1920×1080")
            image = self.np.asarray(source.convert("RGB"))
        result = self.detector.detect(self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=image))
        count = len(result.face_landmarks)
        if count != 1:
            return {"state": "no_face" if count == 0 else "multiple_faces", "faces": count}
        points = result.face_landmarks[0]
        bbox = [max(0, min(p.x for p in points)), max(0, min(p.y for p in points)),
                min(1, max(p.x for p in points)), min(1, max(p.y for p in points))]
        if len(result.facial_transformation_matrixes) != 1:
            return {"state": "pose_unavailable", "faces": 1}
        rotation = self.np.asarray(result.facial_transformation_matrixes[0])[:3, :3]
        if not self.np.isfinite(rotation).all():
            return {"state": "pose_unavailable", "faces": 1}
        # Убираем масштаб матрицы. Порядок Эйлера: Rz(roll) Ry(yaw) Rx(pitch).
        u, _, vt = self.np.linalg.svd(rotation)
        rotation = u @ vt
        if self.np.linalg.det(rotation) < 0:
            u[:, -1] *= -1
            rotation = u @ vt
        pitch = math.degrees(math.atan2(rotation[2, 1], rotation[2, 2]))
        yaw = math.degrees(math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0])))
        roll = math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        return {"state": "face", "faces": 1, "bbox": [round(value, 4) for value in bbox],
                "raw_angles": (pitch, yaw, roll)}

    def close(self):
        self.detector.close()


class AnalysisService:
    """Сборка в потоке сервера, инференс в рабочем потоке, очередь последних кадров."""

    def __init__(self, model_path=None, engine=None):
        self.engine = engine or FaceEngine(model_path)
        self.condition = threading.Condition()
        self.sessions = {}
        self.jobs = OrderedDict()
        self.closed = False
        self.thread = threading.Thread(target=self._work, daemon=True, name="Head analysis")
        self.thread.start()

    def _session(self, code, sid, now):
        session = self.sessions.get(code)
        if session is None or session["sid"] != sid:
            session = {"sid": sid, "assembler": VideoAssembler(), "tracker": MotionTracker(),
                       "result": {"state": "waiting"}, "result_time": now,
                       "submitted": 0, "seen": now, "generation": 0, "last_log": 0, "resets": {}}
            self.sessions[code] = session
        return session

    def control(self, code, role, room, message, now):
        patient = room.get("media", {}).get("patient")
        if patient is None:
            return
        with self.condition:
            session = self._session(code, patient["sid"], now)
            if role == "patient":
                session["seen"] = now
            if message.get("kind") == "bye" and role == "patient":
                self.sessions.pop(code, None)
                self.jobs.pop(code, None)
                return
            camera_off = not patient["camera"]
            reset = message.get("calibration")
            reset_key = (role, message["sid"])
            new_reset = isinstance(reset, str) and len(reset) == 32 and session["resets"].get(reset_key) != reset
            if new_reset:
                session["resets"][reset_key] = reset
            if new_reset or (camera_off and session["result"].get("state") != "camera_off"):
                session["generation"] += 1
                session["tracker"] = MotionTracker()
                session["assembler"] = VideoAssembler()
                session["result"] = {"state": "camera_off" if camera_off else "waiting"}
                session["result_time"] = now
                self.jobs.pop(code, None)

    def video(self, code, sid, message, now):
        with self.condition:
            session = self._session(code, sid, now)
            session["seen"] = now
            jpeg = session["assembler"].add(message, now)
            if jpeg is None or now - session["submitted"] < 0.12:
                return
            session["submitted"] = now
            self.jobs[code] = (session, session["generation"], jpeg, now)
            if len(self.jobs) > 32:
                self.jobs.popitem(last=False)
            self.condition.notify()

    def result(self, code, room, now):
        patient = room.get("media", {}).get("patient")
        if not patient or now - patient["seen"] > PEER_TIMEOUT:
            return {"state": "waiting"}
        if not patient["camera"]:
            return {"state": "camera_off"}
        with self.condition:
            session = self.sessions.get(code)
            if not session or session["sid"] != patient["sid"]:
                return {"state": "waiting"}
            age = now - session["result_time"]
            if age > 2:
                return {"state": "stale"}
            return {**session["result"], "age": round(age, 2)}

    def _work(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or bool(self.jobs), timeout=1)
                if self.closed:
                    return
                now = time.monotonic()
                self.sessions = {k: v for k, v in self.sessions.items() if now - v["seen"] < 10}
                if not self.jobs:
                    continue
                code, (session, generation, jpeg, captured) = self.jobs.popitem(last=False)
                if now - captured > 1:
                    continue
            try:
                face = self.engine.detect(jpeg)
            except Exception:
                face = {"state": "error"}
                logging.getLogger(__name__).exception("Ошибка анализа кадра в комнате %s", code)
            with self.condition:
                if self.sessions.get(code) is not session or session["generation"] != generation:
                    continue
                session["result"] = session["tracker"].update(face, captured)
                session["result_time"] = captured
                if captured - session["last_log"] >= 1:
                    result = session["result"]
                    logging.getLogger(__name__).info(
                        "Комната %s | анализ головы пациента | %s | углы %s | движение %s | кивки %s",
                        code, result["state"], result.get("angles", "—"), result.get("gesture", "—"), result.get("nods", 0))
                    session["last_log"] = captured

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join()
        self.engine.close()
