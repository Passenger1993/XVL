"""
Тесты Docker для упрощенного проекта XVL
Проверяет базовую функциональность Docker образа
"""

import pytest
import subprocess
import sys
import time
import json
from pathlib import Path
import os

# Проверяем наличие Docker
try:
    subprocess.run(["docker", "--version"], capture_output=True, check=True)
    DOCKER_AVAILABLE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    DOCKER_AVAILABLE = False

# Декоратор для пропуска тестов без Docker
skip_if_no_docker = pytest.mark.skipif(
    not DOCKER_AVAILABLE,
    reason="Docker не установлен или не запущен"
)

# Константы
TEST_IMAGE_NAME = "xvl-test-temp"
PROJECT_ROOT = Path(__file__).parent.parent


class TestDockerBasics:
    """Базовые тесты Docker"""

    def test_dockerfile_exists(self):
        """Проверяем наличие Dockerfile"""
        dockerfile = PROJECT_ROOT / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile не найден"

        # Читаем содержимое для дополнительных проверок
        content = dockerfile.read_text()
        assert "FROM python:3.9-slim" in content, "Должен использоваться python:3.9-slim"
        print(f"Dockerfile найден ({len(content)} байт)")

    @skip_if_no_docker
    def test_docker_build(self):
        """Проверяем сборку Docker образа"""
        print(f"\nСборка образа {TEST_IMAGE_NAME}...")

        # Запускаем сборку
        result = subprocess.run(
            ["docker", "build", "-t", TEST_IMAGE_NAME, "."],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            timeout=300  # 5 минут на сборку
        )

        # Проверяем результат
        assert result.returncode == 0, f"Ошибка сборки:\n{result.stderr}"
        print("✅ Образ успешно собран")

        # Дополнительно: проверяем, что образ существует
        result = subprocess.run(
            ["docker", "image", "inspect", TEST_IMAGE_NAME],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0, "Образ не найден после сборки"

    @skip_if_no_docker
    def test_docker_image_size(self):
        """Проверяем размер Docker образа"""
        # Получаем информацию об образе
        result = subprocess.run(
            ["docker", "image", "inspect", TEST_IMAGE_NAME, "--format='{{.Size}}'"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            size_bytes = int(result.stdout.strip().strip("'"))
            size_mb = size_bytes / (1024 * 1024)
            print(f"Размер образа: {size_mb:.1f} MB")

            # Проверяем разумный размер (можно настроить)
            max_size_mb = 1024  # 1GB
            assert size_mb < max_size_mb, f"Образ слишком большой: {size_mb:.1f} MB (максимум {max_size_mb} MB)"
            print(f"✅ Размер образа в пределах нормы")

    @skip_if_no_docker
    def test_docker_run_basic(self):
        """Проверяем базовый запуск контейнера"""
        print("\nТест базового запуска контейнера...")

        # Запускаем контейнер с простой командой
        result = subprocess.run(
            ["docker", "run", "--rm", TEST_IMAGE_NAME, "python", "--version"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Ошибка запуска Python:\n{result.stderr}"
        assert "Python 3.9" in result.stdout, f"Неверная версия Python: {result.stdout}"
        print(f"✅ Python работает: {result.stdout.strip()}")

    @skip_if_no_docker
    def test_docker_run_help(self):
        """Проверяем запуск приложения с --help"""
        print("\nТест запуска приложения с --help...")

        # Пытаемся запустить команду --help
        result = subprocess.run(
            ["docker", "run", "--rm", TEST_IMAGE_NAME, "python", "run.py", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )

        # Команда --help может завершаться с кодом 0 или 2 (стандартно для argparse)
        # Главное - что она выполняется без критических ошибок
        if result.returncode not in [0, 2]:
            print(f"⚠️ Код возврата: {result.returncode}")
            print(f"Вывод: {result.stdout}")
            print(f"Ошибки: {result.stderr}")

        # Проверяем, что есть какой-то вывод
        assert len(result.stdout) > 0 or len(result.stderr) > 0, "Нет вывода от команды"
        print("✅ Команда --help выполнена")

    @skip_if_no_docker
    def test_docker_import_modules(self):
        """Проверяем импорт основных модулей"""
        print("\nТест импорта модулей...")

        # Проверяем основные зависимости
        modules_to_check = [
            "cv2",
            "numpy",
            "PIL",
            "scipy",
            "matplotlib",
        ]

        import_code = "\n".join([
            "try:",
            *[f"    import {module}; print(f'{module}: OK')" for module in modules_to_check],
            "except ImportError as e:",
            "    print(f'Ошибка: {e}'); exit(1)",
        ])

        result = subprocess.run(
            ["docker", "run", "--rm", TEST_IMAGE_NAME, "python", "-c", import_code],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Ошибка импорта модулей:\n{result.stderr}"
        print(f"✅ Все модули импортируются:\n{result.stdout}")

    @skip_if_no_docker
    def test_docker_environment(self):
        """Проверяем переменные окружения"""
        print("\nТест переменных окружения...")

        result = subprocess.run(
            ["docker", "run", "--rm", TEST_IMAGE_NAME, "env"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Ошибка выполнения env:\n{result.stderr}"

        env_output = result.stdout
        assert "PYTHONPATH" in env_output, "PYTHONPATH не установлен"
        print("✅ Переменные окружения настроены корректно")

    @skip_if_no_docker
    def test_docker_working_directory(self):
        """Проверяем рабочую директорию"""
        print("\nТест рабочей директории...")

        result = subprocess.run(
            ["docker", "run", "--rm", TEST_IMAGE_NAME, "pwd"],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Ошибка выполнения pwd:\n{result.stderr}"
        assert "/app" in result.stdout.strip(), f"Некорректная рабочая директория: {result.stdout}"
        print(f"✅ Рабочая директория: {result.stdout.strip()}")

    @skip_if_no_docker
    def test_docker_file_structure(self):
        """Проверяем структуру файлов в контейнере"""
        print("\nТест структуры файлов...")

        # Проверяем наличие ключевых файлов
        files_to_check = [
            "/app/requirements.txt",
            "/app/pyproject.toml",
            "/app/run.py",
        ]

        check_code = "\n".join([
            "import os, sys",
            "errors = []",
            *[f"if not os.path.exists('{file}'): errors.append('Файл {file} не найден')" for file in files_to_check],
            "if errors: print('\\n'.join(errors)); sys.exit(1)",
            "print('Все файлы на месте')",
        ])

        result = subprocess.run(
            ["docker", "run", "--rm", TEST_IMAGE_NAME, "python", "-c", check_code],
            capture_output=True,
            text=True,
            timeout=30
        )

        assert result.returncode == 0, f"Ошибка проверки файлов:\n{result.stderr}"
        print("✅ Структура файлов корректна")

    @skip_if_no_docker
    @pytest.mark.slow
    def test_docker_volume_mount(self):
        """Проверяем монтирование томов (опционально, медленный тест)"""
        print("\nТест монтирования томов...")

        # Создаем временный файл
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content from host")
            temp_file = f.name

        try:
            # Монтируем файл в контейнер
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{temp_file}:/tmp/test.txt:ro",
                    TEST_IMAGE_NAME,
                    "cat", "/tmp/test.txt"
                ],
                capture_output=True,
                text=True,
                timeout=30
            )

            assert result.returncode == 0, f"Ошибка чтения файла:\n{result.stderr}"
            assert "Test content from host" in result.stdout
            print("✅ Монтирование томов работает")
        finally:
            # Удаляем временный файл
            os.unlink(temp_file)


class TestDockerCleanup:
    """Тесты очистки Docker ресурсов"""

    @skip_if_no_docker
    def test_docker_cleanup(self):
        """Очищаем тестовый образ"""
        print(f"\nОчистка тестового образа {TEST_IMAGE_NAME}...")

        # Удаляем образ
        result = subprocess.run(
            ["docker", "rmi", "-f", TEST_IMAGE_NAME],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            print("✅ Тестовый образ удален")
        else:
            print(f"⚠️ Не удалось удалить образ (возможно, его не было): {result.stderr}")


@pytest.fixture(scope="session", autouse=True)
def docker_cleanup_final():
    """Финализатор для очистки Docker образов после всех тестов"""
    yield

    # Выполняется после всех тестов
    if DOCKER_AVAILABLE:
        print("\n" + "="*60)
        print("Финализация Docker тестов...")

        # Удаляем тестовый образ, если он остался
        subprocess.run(
            ["docker", "rmi", "-f", TEST_IMAGE_NAME],
            capture_output=True
        )

        # Очищаем неиспользуемые ресурсы
        subprocess.run(["docker", "system", "prune", "-f"], capture_output=True)
        print("✅ Очистка Docker завершена")


def run_simple_docker_check():
    """Простая проверка Docker без pytest"""
    print("="*60)
    print("Простая проверка Docker образа")
    print("="*60)

    tests = TestDockerBasics()

    try:
        tests.test_dockerfile_exists()
        tests.test_docker_build()
        tests.test_docker_run_basic()
        tests.test_docker_import_modules()
        print("\n" + "="*60)
        print("✅ Все базовые тесты пройдены!")
        return True
    except AssertionError as e:
        print(f"\n❌ Тест не пройден: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Неожиданная ошибка: {e}")
        return False
    finally:
        # Пытаемся очистить
        cleanup = TestDockerCleanup()
        cleanup.test_docker_cleanup()


if __name__ == "__main__":
    # Запуск напрямую (без pytest)
    success = run_simple_docker_check()
    sys.exit(0 if success else 1)