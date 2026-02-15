# Dockerfile
FROM python:3.9-slim as builder

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости
COPY requirements.txt .
COPY pyproject.toml .

# Создаем виртуальное окружение и устанавливаем зависимости
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==== Финальный образ ====
FROM python:3.9-slim

# Создаем не-root пользователя
RUN groupadd -r xvl && useradd -r -g xvl -u 1000 xvl

WORKDIR /app

# Копируем виртуальное окружение из builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Копируем исходный код
COPY . .

# Настройка прав
RUN chown -R xvl:xvl /app
USER xvl

# Проверка здоровья
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Команда по умолчанию
CMD ["python", "run.py", "--help"]

# Метаданные
LABEL maintainer="your-team@example.com"
LABEL version="1.0"
LABEL description="XVL - X-Ray Vision Lab"