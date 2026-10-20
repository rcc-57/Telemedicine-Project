"""UDP-сервер комнат, звонков и анализа головы пациента."""

import argparse
import logging
import sys
from pathlib import Path

from Lab_Work_two import UDPServer
from head_analysis import AnalysisService


def main():
    parser = argparse.ArgumentParser(description="UDP-сервер с анализом головы")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model", type=Path, default=Path(__file__).parent / "models" / "face_landmarker.task")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S", stream=sys.stdout)
    if not args.model.is_file():
        parser.error("Нет модели. Выполните .venv/bin/python download_head_model.py")
    try:
        analysis = AnalysisService(args.model)
    except (ImportError, OSError, RuntimeError) as error:
        parser.error(f"Не удалось загрузить анализатор: {error}. Установите requirements-server.txt")
    try:
        with UDPServer((args.host, args.port)) as server:
            server.analysis = analysis
            print(f"Сервер анализа головы запущен: {args.host}:{args.port}", flush=True)
            print("Анализируется камера пациента. Для калибровки смотрите прямо 2 секунды.", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                print("\nСервер остановлен")
    except OSError as error:
        parser.error(f"Не удалось открыть UDP-порт: {error}. Остановите прежний сервер на этом порту.")
    finally:
        analysis.close()


if __name__ == "__main__":
    main()
