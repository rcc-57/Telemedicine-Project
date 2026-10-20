import base64
import threading
import time
import unittest
from unittest.mock import Mock

from head_analysis import AnalysisService, MotionTracker
from Lab_Work_two import Application
from media import pack, relay, unpack
from VideoCall import analysis_text


def face(pitch=0, yaw=0, roll=0):
    return {"state": "face", "faces": 1, "bbox": [0.2, 0.1, 0.8, 0.9],
            "raw_angles": (pitch, yaw, roll)}


class MotionTests(unittest.TestCase):
    def setUp(self):
        self.tracker = MotionTracker()
        self.now = 0
        for _ in range(10):
            self.step()

    def step(self, pitch=0, yaw=0, roll=0):
        self.now += 0.15
        return self.tracker.update(face(pitch, yaw, roll), self.now)

    def test_stationary_and_relative_pose(self):
        for _ in range(20):
            result = self.step()
        self.assertEqual(result["state"], "tracking")
        self.assertFalse(result["moving"])
        self.assertEqual((result["nods"], result["shakes"]), (0, 0))
        for _ in range(4):
            result = self.step(20, 25, -20)
        self.assertEqual((result["turn"], result["vertical"], result["tilt"]), ("right", "down", "right"))

    def test_nod_and_shake(self):
        for pitch in (8, 18, 24, 18, 8, 0, 0, 0):
            result = self.step(pitch=pitch)
        self.assertEqual(result["nods"], 1)
        self.assertEqual(result["gesture"], "nod")
        for yaw in (10, 22, 28, 15, 0, -15, -28, -28, -15, 0, 0, 0):
            result = self.step(yaw=yaw)
        self.assertEqual(result["shakes"], 1)
        self.assertEqual(result["gesture"], "shake")

    def test_face_loss_and_multiple_faces_reset_calibration(self):
        result = self.tracker.update({"state": "no_face", "faces": 0}, self.now + 0.1)
        self.assertEqual(result["state"], "no_face")
        self.assertEqual(self.step()["state"], "calibrating")
        result = self.tracker.update({"state": "multiple_faces", "faces": 2}, self.now + 0.1)
        self.assertNotIn("angles", result)
        self.assertIsNone(self.tracker.baseline)
        self.assertIn("несколько лиц", analysis_text(result))

    def test_moving_during_calibration_does_not_set_baseline(self):
        tracker = MotionTracker()
        for index in range(30):
            result = tracker.update(face(yaw=10 if index % 2 else -10), index * 0.15)
        self.assertEqual(result["state"], "calibrating")


class ServiceTests(unittest.TestCase):
    def setUp(self):
        self.app = Application()
        doctor = self.app.dispatch({"method": "POST", "path": "/rooms"})[1]
        self.code = doctor["room"]
        patient = self.app.dispatch({"method": "POST", "path": f"/rooms/{self.code}/join"})[1]
        self.doctor = dict(room=self.code, token=doctor["token"], sid="d" * 32)
        self.patient = dict(room=self.code, token=patient["token"], sid="a" * 32)
        self.engine = Mock()
        self.engine.detect.return_value = face()
        self.service = AnalysisService(engine=self.engine)
        self.now = time.monotonic()

    def tearDown(self):
        self.service.close()

    def send(self, identity=None, **message):
        return relay(self.app.rooms, pack({**(identity or self.patient), **message}),
                     ("127.0.0.1", 12345), self.now, analysis=self.service)

    def test_patient_only_auth_and_analysis_without_peer_call(self):
        self.send(kind="hello", camera=True, microphone=False)
        self.send(self.doctor, kind="hello", camera=True, microphone=False)
        data = dict(kind="video", seq=1, index=0, total=1, data=base64.b64encode(b"jpeg").decode())
        self.send(self.doctor, **data)
        self.assertFalse(self.engine.detect.called)
        bad = {**self.patient, "token": "wrong"}
        self.send(bad, **data)
        self.assertFalse(self.engine.detect.called)
        self.send(**data)
        deadline = time.monotonic() + 1
        while not self.engine.detect.called and time.monotonic() < deadline:
            time.sleep(0.01)
        self.engine.detect.assert_called_once_with(b"jpeg")
        with self.service.condition:
            self.assertEqual(self.service.sessions[self.code]["result"]["state"], "calibrating")
        response = unpack(self.send(kind="hello", camera=True, microphone=False)[0][0])
        self.assertEqual(response["analysis"]["state"], "calibrating")
        self.send(kind="hello", camera=False, microphone=False)
        self.assertEqual(self.service.result(self.code, self.app.rooms[self.code], self.now)["state"], "camera_off")

    def test_calibration_request_is_idempotent(self):
        self.send(kind="hello", camera=True, microphone=False, calibration="b" * 32)
        generation = self.service.sessions[self.code]["generation"]
        self.send(kind="hello", camera=True, microphone=False, calibration="b" * 32)
        self.assertEqual(self.service.sessions[self.code]["generation"], generation)
        self.send(kind="hello", camera=True, microphone=False, calibration="c" * 32)
        self.assertEqual(self.service.sessions[self.code]["generation"], generation + 1)

    def test_late_result_after_camera_off_is_discarded(self):
        started, finish = threading.Event(), threading.Event()
        def detect(data):
            started.set()
            finish.wait(2)
            return face()
        self.engine.detect.side_effect = detect
        self.send(kind="hello", camera=True, microphone=False)
        self.send(kind="video", seq=1, index=0, total=1, data="eA==")
        self.assertTrue(started.wait(1))
        self.send(kind="hello", camera=False, microphone=False)
        finish.set()
        self.service.close()
        self.assertEqual(self.service.sessions[self.code]["result"]["state"], "camera_off")


if __name__ == "__main__":
    unittest.main()
