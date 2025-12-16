# Upload to Hugging Face Hub
from huggingface_hub import HfApi, create_repo
from pathlib import Path

def upload_model(model_path: str, repo_name: str, token: str):
    """Загрузка модели на Hugging Face Hub"""
    api = HfApi()

    # Создаём репозиторий если его нет
    try:
        create_repo(repo_name, token=token, exist_ok=True)
    except:
        pass

    # Загружаем модель
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model.pt",
        repo_id=repo_name,
        repo_type="model"
    )

    # Загружаем конфиг
    config_path = Path(model_path).parent / "config.yaml"
    if config_path.exists():
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.yaml",
            repo_id=repo_name,
            repo_type="model"
        )