from PIL import Image, ImageDraw, ImageFilter
import random
import time, datetime, tqdm
from generate_seam import draw_a_seam

def make_empty_seam(image):
	"""
	Создает изображение с непроваром
	"""
	result = image.copy()
	draw = ImageDraw.Draw(result)
	width, height = image.size

	# Рисуем шов и получаем центры чешуек и граничные точки
	_ , _, _, _ = draw_a_seam(draw, width, height)

	return result

def generate_dataset(directory, num_images=100):
	progress_bar = tqdm(total=num_images)
	for i in range(num_images):
		empty_image = make_empty_seam()
		empty_image.save(f"{directory}/E{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
		progress_bar.set_description(f"Генерация глухарей № {i+1}")
		progress_bar.update(1)



