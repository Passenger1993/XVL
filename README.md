# X-Ray Vision Lab (XVL)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Нейросетевая модель для автоматического анализа рентгеновских снимков сварных швов металлоконструкций на предмет дефектов.
Этот проект использует **синтетическую генерацию данных** на лету. 
## 🚀 Быстрый старт

### Установка


```bash
# Клонируем репозиторий
git clone https://github.com/yourname/XVL.git
cd XVL

# Устанавливаем зависимости
pip install -r requirements.txt

# Загружаем предобученную модель
python -m src.model.download --model xvl-v1.0