from PIL import Image, ImageDraw, ImageFilter
import random
import time, datetime
from tqdm import tqdm
from src.generators.generate_seam import draw_a_seam
import os

def make_empty_seam():
	"""
	Создает изображение с непроваром
	"""

	result, draw, centers, direction, start_point, end_point, pattern_size, seam_area = draw_a_seam()

	return result

def generate_dataset(directory, num_images=100):
    # Создаем директорию, если она не существует
    os.makedirs(directory, exist_ok=True)

    progress_bar = tqdm(total=num_images)
    for i in range(num_images):
        empty_image = make_empty_seam()

        # Формируем имя файла
        filename = f"E{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
        file_path = os.path.join(directory, filename)

        # Сохраняем изображение
        empty_image.save(file_path)
        progress_bar.set_description(f"Генерация глухарей № {i+1}")
        progress_bar.update(1)

    progress_bar.close()