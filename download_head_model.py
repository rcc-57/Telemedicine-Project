import hashlib
from pathlib import Path
from urllib.request import urlopen

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"


def main():
    destination = Path(__file__).parent / "models" / "face_landmarker.task"
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == SHA256:
        print(f"Модель уже готова: {destination}")
        return
    with urlopen(URL, timeout=60) as response:
        data = response.read(20 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Контрольная сумма модели не совпала; файл не сохранён")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(destination)
    print(f"Модель сохранена: {destination}")


if __name__ == "__main__":
    main()
