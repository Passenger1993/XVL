FROM python:3.9-slim

WORKDIR /app

# Копируем только необходимое
COPY requirements.txt .
COPY pyproject.toml .

# Установка зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код
COPY . .

# Команда по умолчанию
CMD ["python", "run.py", "--help"]