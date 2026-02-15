.PHONY: help install install-dev test test-unit test-integration coverage lint format clean run

# Цвета для вывода
GREEN=\033[0;32m
RED=\033[0;31m
NC=\033[0m

help: ## Показать справку по командам
	@echo "$(GREEN)Доступные команды:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: ## Установить проект и зависимости
	@echo "$(GREEN)Установка проекта...$(NC)"
	pip install -e .

install-dev: ## Установить зависимости для разработки
	@echo "$(GREEN)Установка зависимостей для разработки...$(NC)"
	pip install -e ".[dev]"
	pre-commit install

test: ## Запустить все тесты
	@echo "$(GREEN)Запуск всех тестов...$(NC)"
	pytest tests/ -v --tb=short -m "not slow"

test-unit: ## Запустить unit-тесты
	@echo "$(GREEN)Запуск unit-тестов...$(NC)"
	pytest tests/unit/ -v --tb=short -m "unit"

test-integration: ## Запустить интеграционные тесты
	@echo "$(GREEN)Запуск интеграционных тестов...$(NC)"
	pytest tests/integration/ -v --tb=short -m "integration"

test-slow: ## Запустить медленные тесты
	@echo "$(GREEN)Запуск медленных тестов...$(NC)"
	pytest tests/ -v --tb=short -m "slow"

test-evaluator: ## Запустить тесты evaluator
	@echo "$(GREEN)Запуск тестов evaluator...$(NC)"
	pytest tests/ -k "evaluator" -v --tb=short

test-manager: ## Запустить тесты manager
	@echo "$(GREEN)Запуск тестов manager...$(NC)"
	pytest tests/ -k "manager" -v --tb=short

test-zippacker: ## Запустить тесты zip_packer
	@echo "$(GREEN)Запуск тестов zip_packer...$(NC)"
	pytest tests/ -k "zip_packer" -v --tb=short

test-generators: ## Запустить тесты генераторов
	@echo "$(GREEN)Запуск тестов генераторов...$(NC)"
	pytest tests/ -m "generator" -v --tb=short

coverage: ## Запустить тесты с измерением покрытия
	@echo "$(GREEN)Запуск тестов с покрытием...$(NC)"
	pytest tests/ \
		--cov=. \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-config=pyproject.toml \
		--tb=short
	@echo "$(GREEN)Отчет о покрытии: htmlcov/index.html$(NC)"

coverage-badge: ## Создать бейдж покрытия
	@echo "$(GREEN)Создание бейджа покрытия...$(NC)"
	coverage-badge -o coverage.svg

lint: ## Проверить код стилем
	@echo "$(GREEN)Проверка стиля кода...$(NC)"
	black --check .
	flake8 .
	mypy .
	bandit -r . -c pyproject.toml

format: ## Автоформатирование кода
	@echo "$(GREEN)Форматирование кода...$(NC)"
	black .
	isort .

pre-commit-all: ## Запустить pre-commit для всех файлов
	@echo "$(GREEN)Запуск pre-commit...$(NC)"
	pre-commit run --all-files

clean: ## Очистить временные файлы
	@echo "$(GREEN)Очистка временных файлов...$(NC)"
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .hypothesis/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete

run: ## Запустить приложение
	@echo "$(GREEN)Запуск приложения...$(NC)"
	python -m src.main

docker-test: ## Запустить тесты в Docker
	@echo "$(GREEN)Запуск тестов в Docker...$(NC)"
	docker build -t xvl-test -f Dockerfile.test .
	docker run --rm xvl-test

benchmark: ## Запустить бенчмарк-тесты
	@echo "$(GREEN)Запуск бенчмарк-тестов...$(NC)"
	pytest tests/ -m "benchmark" -v --benchmark-only

property-test: ## Запустить property-based тесты
	@echo "$(GREEN)Запуск property-based тестов...$(NC)"
	pytest tests/ -m "property" -v

ci-test: ## Запуск тестов для CI/CD (без GUI)
	@echo "$(GREEN)Запуск тестов для CI/CD...$(NC)"
	pytest tests/ \
		-v \
		--tb=short \
		--strict-markers \
		--disable-warnings \
		-m "not gui and not requires_gpu" \
		--junitxml=test-results.xml \
		--cov=. \
		--cov-report=xml

update-deps: ## Обновить зависимости
	@echo "$(GREEN)Обновление зависимостей...$(NC)"
	pip install --upgrade pip
	pip install --upgrade -e ".[dev]"

security-check: ## Проверка безопасности
	@echo "$(GREEN)Проверка безопасности...$(NC)"
	bandit -r . -c pyproject.toml -f html -o security-report.html
	safety check

docs: ## Создать документацию
	@echo "$(GREEN)Создание документации...$(NC)"
	pdoc --html --output-dir docs/ .

.PHONY: docker-build docker-run docker-test docker-clean

docker-build: ## Собрать Docker образ
	@echo "$(GREEN)Сборка Docker образа...$(NC)"
	docker build -t xvl:latest .

docker-run: ## Запустить Docker контейнер (базовая проверка)
	@echo "$(GREEN)Запуск Docker контейнера...$(NC)"
	docker run --rm xvl:latest python --version

docker-run-help: ## Запустить приложение с --help
	@echo "$(GREEN)Запуск приложения с --help...$(NC)"
	docker run --rm xvl:latest python run.py --help || true

docker-test: ## Запустить тесты Docker
	@echo "$(GREEN)Запуск тестов Docker...$(NC)"
	SKIP_DOCKER_TESTS=false pytest tests/test_docker.py -v --tb=short

docker-test-quick: ## Быстрые тесты Docker (без медленных тестов)
	@echo "$(GREEN)Запуск быстрых тестов Docker...$(NC)"
	SKIP_DOCKER_TESTS=false pytest tests/test_docker.py -v --tb=short -m "not slow"

docker-test-manual: ## Ручная проверка Docker (без pytest)
	@echo "$(GREEN)Ручная проверка Docker...$(NC)"
	python tests/test_docker.py

docker-clean: ## Очистить Docker ресурсы
	@echo "$(GREEN)Очистка Docker ресурсов...$(NC)"
	docker system prune -f
	docker images | grep xvl | awk '{print $$3}' | xargs -r docker rmi -f 2>/dev/null || true

# Интеграция в общие команды
test-all: test-unit test-integration docker-test-quick ## Все тесты (включая быстрые Docker тесты)

ci-docker-test: ## Docker тесты для CI/CD
	@echo "$(GREEN)Docker тесты для CI/CD...$(NC)"
	# Сборка
	docker build -t xvl-ci:latest .
	# Базовые проверки
	docker run --rm xvl-ci:latest python --version
	docker run --rm xvl-ci:latest python -c "import cv2, numpy, PIL, scipy, matplotlib; print('Импорт модулей OK')"
	# Очистка
	docker rmi xvl-ci:latest