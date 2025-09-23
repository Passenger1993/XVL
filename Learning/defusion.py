from PIL import Image, ImageDraw, ImageFilter
import random
import math
import datetime
from generate_seam import draw_a_seam
import statistics
import json
from tqdm import tqdm
import time


def _draw_defect_centered(draw, centers, defect_width, accuracy=0.9):
	"""
	Рисует непровар вдоль шва с эффектом истёртости и плавными переходами

	Args:
		draw: объект для рисования
		centers: список координат центров чешуек шва
		defect_width: базовая ширина непровара
		accuracy: точность выполнения (0.0-1.0), где 1.0 - идеальная линия
	"""
	if not centers or len(centers) < 2:
		return

	# Вычисляем общую длину пути через все центры
	total_length = 0
	segment_lengths = []
	for i in range(len(centers) - 1):
		dx = centers[i + 1][0] - centers[i][0]
		dy = centers[i + 1][1] - centers[i][1]
		length = math.sqrt(dx * dx + dy * dy)
		segment_lengths.append(length)
		total_length += length

	if total_length == 0:
		return

	# Создаём контрольные точки для плавного шума
	num_control_points = max(3, int(total_length / 5))
	control_points = [random.random() for _ in range(num_control_points)]

	# Функция для получения значения шума в позиции t (0-1)
	def get_noise(t):
		idx = t * (num_control_points - 1)
		idx0 = int(idx)
		idx1 = min(num_control_points - 1, idx0 + 1)
		frac = idx - idx0
		return control_points[idx0] * (1 - frac) + control_points[idx1] * frac

	# Рисуем непровар с эффектом истёртости вдоль всего пути
	step = random.randint(2, 5)
	prev_draw = False
	prev_x, prev_y = centers[0]

	current_distance = 0
	for i in range(len(centers) - 1):
		start_x, start_y = centers[i]
		end_x, end_y = centers[i + 1]

		dx = end_x - start_x
		dy = end_y - start_y
		segment_length = segment_lengths[i]

		if segment_length == 0:
			continue

		# Нормализуем направляющий вектор для этого сегмента
		dx_norm = dx / segment_length
		dy_norm = dy / segment_length

		# Проходим по текущему сегменту
		for distance in range(0, int(segment_length) + 1, step):
			t_segment = distance / segment_length
			x = start_x + t_segment * dx
			y = start_y + t_segment * dy

			# Общая позиция вдоль всего пути (0-1)
			t_total = (current_distance + distance) / total_length

			# Получаем значение шума для этой позиции
			noise = get_noise(t_total) - random.random() / 10

			# Определяем, нужно ли рисовать в этой позиции
			should_draw = noise < accuracy

			# Определяем толщину в этой позиции
			if should_draw:
				# Толщина зависит от accuracy и значения шума
				thickness_factor = 0.8 + 0.8 * (noise / accuracy) if accuracy > 0 else 0
				current_width = max(1, defect_width * thickness_factor)

				# Добавляем небольшие неровности к толщине
				current_width *= random.uniform(0.8, 1.2)

				# Рисуем сегмент
				if prev_draw:
					# Плавный переход от предыдущего сегмента
					draw.line([(prev_x, prev_y), (x, y)], fill=random.randint(0, 20),
							  width=int(current_width), joint="curve")
				else:
					# Начинаем новый сегмент
					radius = current_width / 4
					draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=random.randint(0, 20))
			elif prev_draw:
				# Завершаем предыдущий сегмент
				radius = defect_width / 2
				draw.ellipse([prev_x - radius, prev_y - radius, prev_x + radius, prev_y + radius],
							 fill=random.randint(0, 20))

			prev_draw = should_draw
			prev_x, prev_y = x, y

		current_distance += segment_length

	# Завершаем последний сегмент, если нужно
	if prev_draw:
		radius = defect_width / 2
		draw.ellipse([prev_x - radius, prev_y - radius, prev_x + radius, prev_y + radius], fill=random.randint(0, 20))
	central_x = statistics.mean([x for x,y in centers])
	central_y = statistics.mean([y for x,y in centers])
	spread_x = max([x-central_x for x,y in centers], key=abs)
	spread_y = max([y-central_y for x,y in centers], key=abs)
	target_area = [central_x + spread_x, central_y + spread_y, central_x - spread_x, central_y - spread_y]
	return {"defusion":target_area}

def make_incomplete_fusion():
	# Рисуем шов и получаем центры чешуек и граничные точки
	result, draw, centers, direction, start_point, end_point, pattern_size = draw_a_seam()

	# Параметры непровара
	defect_width = random.randint(2, 4)

	# Рисуем непровар вдоль шва
	data = _draw_defect_centered(draw, centers, defect_width, 0.9)

	return result, data


def generate_dataset(directory, num_images=100):
	summary = []
	progress_bar = tqdm(total=num_images)
	for i in range(num_images):
		fusion_image, data = make_incomplete_fusion()
		fusion_image.save(f"{directory}/N{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
		summary.append(data)
		progress_bar.set_description(f"Генерация непровара № {i+1}")
		progress_bar.update(1)
	with open(f"{directory}/summary.jsom","w") as file:
		json.dump(summary, file)
