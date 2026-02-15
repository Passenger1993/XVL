# Добавить в существующий conftest.py
import pytest
import subprocess
import sys

@pytest.fixture(scope="session")
def docker_available():
    """Проверяем, доступен ли Docker"""
    try:
        subprocess.run(["docker", "--version"],
                      capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

@pytest.fixture
def skip_if_no_docker(docker_available):
    """Пропускаем тест, если Docker недоступен"""
    if not docker_available:
        pytest.skip("Docker не установлен или не запущен")

@pytest.fixture
def docker_image(docker_available, tmp_path_factory):
    """Фикстура для тестового Docker образа"""
    if not docker_available:
        pytest.skip("Docker недоступен")

    import tempfile
    import shutil

    # Создаем временную директорию для сборки
    temp_dir = tmp_path_factory.mktemp("docker_build")

    # Копируем необходимые файлы
    files_to_copy = [
        "Dockerfile",
        "requirements.txt",
        "pyproject.toml",
        "run.py",
    ]

    for file_name in files_to_copy:
        src = Path(__file__).parent.parent / file_name
        if src.exists():
            shutil.copy2(src, temp_dir / file_name)

    # Собираем образ
    image_name = "xvl-test-fixture"
    result = subprocess.run(
        ["docker", "build", "-t", image_name, "."],
        cwd=str(temp_dir),
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        pytest.fail(f"Не удалось собрать Docker образ: {result.stderr}")

    yield image_name

    # Очистка после тестов
    subprocess.run(
        ["docker", "rmi", "-f", image_name],
        capture_output=True
    )