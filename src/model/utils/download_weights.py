# scripts/download_weights.py

import os
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

def download_model_weights(
    repo_id: str = "Alex-Watchman/XVL",  # ЗАМЕНИТЕ на свой
    filename: str = "best.pt",
    local_dir: Path = Path(__file__).parent.parent / "src" / "model" / "weights"
) -> Path:
    """
    Скачивает файл весов модели с Hugging Face Hub в указанную локальную директорию.

    Args:
        repo_id: Идентификатор репозитория на HF (например, 'ultralytics/yolov5').
        filename: Название файла для скачивания.
        local_dir: Локальная папка для сохранения.

    Returns:
        Полный путь к скачанному файлу.
    """
    try:
        # 1. Создаём целевую директорию, если её нет
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"🟡 Целевая директория: {local_dir.absolute()}")

        # 2. Проверяем, существует ли файл уже (можно пропустить)
        local_path = local_dir / filename
        if local_path.exists():
            print(f"✅ Файл '{filename}' уже существует. Скачивание пропущено.")
            return local_path

        # 3. Скачиваем файл с Hub
        print(f"⬇️  Скачивание '{filename}' из репозитория '{repo_id}'...")
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Копируем файл, а не создаём симлинк
            resume_download=True           # Продолжаем при разрыве соединения
        )

        print(f"✅ Файл успешно скачан: {downloaded_path}")
        return Path(downloaded_path)

    except Exception as e:
        print(f"❌ Ошибка при скачивании: {e}", file=sys.stderr)
        sys.exit(1)  # Выходим с кодом ошибки

if __name__ == "__main__":
    target_dir = Path(__file__).parent.parent / "weights"
    download_model_weights(local_dir=target_dir)