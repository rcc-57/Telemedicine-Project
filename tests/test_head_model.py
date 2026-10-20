"""Необязательная проверка настоящей модели с тестовым портретом HEAD_TEST_IMAGE."""
import io
import os
import threading
import time
import unittest


@unittest.skipUnless(os.environ.get("HEAD_TEST_IMAGE"), "Нужен тестовый портрет HEAD_TEST_IMAGE")
class RealModelTests(unittest.TestCase):
    def test_model_and_udp_analysis(self):
        from PIL import Image
        from head_analysis import FaceEngine, AnalysisService
        from Lab_Work_two import UDPServer
        from Client import Client
        from media import MediaConnection

        def jpeg(image):
            output = io.BytesIO()
            image.convert("RGB").save(output, format="JPEG")
            return output.getvalue()

        portrait = Image.open(os.environ["HEAD_TEST_IMAGE"])
        portrait.thumbnail((480, 360))
        data = jpeg(portrait)
        engine = FaceEngine("models/face_landmarker.task")
        self.assertEqual(engine.detect(jpeg(Image.new("RGB", (480, 360))))["state"], "no_face")
        neutral = engine.detect(data)
        self.assertEqual(neutral["faces"], 1)
        tilted = engine.detect(jpeg(portrait.rotate(15)))
        self.assertEqual(tilted["faces"], 1)
        self.assertGreater(tilted["raw_angles"][2] - neutral["raw_angles"][2], 8)
        analysis = AnalysisService(engine=engine)
        server = UDPServer(("127.0.0.1", 0))
        server.analysis = analysis
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        doctor = Client(*server.server_address)
        patient = Client(*server.server_address)
        video = None
        try:
            doctor.enter()
            patient.enter(doctor.room)
            video = MediaConnection(server.server_address, patient.room, patient.token)
            video.controls(camera=True)
            time.sleep(0.3)
            for _ in range(20):
                video.send_video(data)
                time.sleep(0.15)
            deadline = time.monotonic() + 2
            while video.analysis.get("state") != "tracking" and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(video.analysis["state"], "tracking", video.analysis)
            self.assertEqual(video.analysis["turn"], "center")
            self.assertEqual(video.analysis["nods"], 0)
            video.controls(camera=False)
            deadline = time.monotonic() + 2
            while video.analysis.get("state") != "camera_off" and time.monotonic() < deadline:
                time.sleep(0.05)
            self.assertEqual(video.analysis["state"], "camera_off")
        finally:
            if video is not None:
                video.close()
            doctor.close()
            patient.close()
            server.shutdown()
            thread.join()
            server.server_close()
            analysis.close()


if __name__ == "__main__":
    unittest.main()
