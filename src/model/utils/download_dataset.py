#!/usr/bin/env python3
"""
Скрипт для загрузки и распаковки ZIP-архива с датасетом в папку ./data/training.
Пример использования:
    python download_dataset.py https://huggingface.co/datasets/username/dataset/resolve/main/data.zip
"""

import os
import sys
import argparse
import requests
import zipfile
import tempfile
import shutil
from pathlib import Path

def download_file(url, local_path, chunk_size=8192):
    """Скачивает файл по URL с отображением прогресса."""
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            downloaded = 0
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = downloaded / total_size * 100
                        print(f"\rПрогресс: {percent:.1f}% ({downloaded}/{total_size} байт)", end='')
                    else:
                        print(f"\rСкачано {downloaded} байт...", end='')
            print()  # новая строка после завершения
    except Exception as e:
        print(f"\nОшибка при скачивании: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Скачивание и распаковка датасета")
    parser.add_argument("url", help="Прямая ссылка на ZIP-архив", default="https://huggingface.co/datasets/Alex-Watchman/XVL_train/train.zip")
    parser.add_argument("--no-unzip", action="store_true", help="Не распаковывать архив, только скачать")
    parser.add_argument("--target", default="./data/training", help="Целевая папка (по умолчанию ./data/training)")
    args = parser.parse_args()

    target_dir = Path(args.target)
    # Создаём папку, если её нет
    target_dir.mkdir(parents=True, exist_ok=True)

    # Имя файла из URL или заданное
    filename = os.path.basename(args.url.split('?')[0])  # убираем query-параметры
    if not filename.endswith('.zip'):
        filename = 'train.zip'  # запасное имя

    temp_zip = target_dir / filename

    print(f"📥 Скачивание {args.url} -> {temp_zip}")
    try:
        download_file(args.url, temp_zip)
    except Exception:
        sys.exit(1)

    if not args.no_unzip:
        print(f"📦 Распаковка {temp_zip} в {target_dir}...")
        try:
            with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
            print("✅ Распаковка завершена.")
            # Удаляем архив после распаковки
            temp_zip.unlink()
            print(f"🗑️ Архив {temp_zip} удалён.")
        except zipfile.BadZipFile:
            print(f"❌ Ошибка: файл {temp_zip} не является корректным ZIP-архивом.")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Ошибка при распаковке: {e}")
            sys.exit(1)
    else:
        print(f"✅ Архив сохранён как {temp_zip} (без распаковки).")

if __name__ == "__main__":
    main()