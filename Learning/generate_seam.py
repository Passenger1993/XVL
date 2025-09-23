from PIL import Image, ImageDraw, ImageFilter
import random, math
import numpy as np

import math

def create_parallel_line(x1, y1, x2, y2, offset, length=None):
    """
    Создает параллельную линию со смещением

    Args:
        x1, y1, x2, y2: координаты исходной линии
        offset: смещение от исходной линии (положительное - одна сторона, отрицательное - другая)
        length: длина линии в обе стороны от центра. Если None - сохраняет исходную длину

    Returns:
        Координаты параллельной линии (x1, y1, x2, y2)
    """
    # Вычисляем вектор направления линии
    dx = x2 - x1
    dy = y2 - y1

    # Вычисляем длину исходной линии
    original_length = math.sqrt(dx*dx + dy*dy)

    if original_length == 0:
        return (x1, y1, x2, y2)

    # Нормализуем вектор направления
    dx_norm = dx / original_length
    dy_norm = dy / original_length

    # Вычисляем перпендикулярный вектор (поворот на 90 градусов)
    perp_dx = -dy_norm
    perp_dy = dx_norm

    # Смещаем обе точки по перпендикуляру
    new_x1 = x1 + perp_dx * offset
    new_y1 = y1 + perp_dy * offset
    new_x2 = x2 + perp_dx * offset
    new_y2 = y2 + perp_dy * offset

    # Если указана новая длина, расширяем линию в обе стороны
    if length is not None:
        # Вычисляем середину линии
        mid_x = (new_x1 + new_x2) / 2
        mid_y = (new_y1 + new_y2) / 2

        # Половина новой длины
        half_length = length / 2

        # Создаем новую линию нужной длины
        new_x1 = mid_x - dx_norm * half_length
        new_y1 = mid_y - dy_norm * half_length
        new_x2 = mid_x + dx_norm * half_length
        new_y2 = mid_y + dy_norm * half_length

    return (new_x1, new_y1, new_x2, new_y2)

def draw_a_seam(pattern_size=random.randint(15, 30), overlap_ratio=random.randint(12, 20) // 10, kernel_div=3):
	"""
	Рисует сварной шов из повторяющихся узоров в виде чешуек.
	Возвращает центры чешуек для последующего рисования непровара.

	Returns:
		centers: список координат центров чешуек
		direction: направление шва
		start_point: начальная точка шва
		end_point: конечная точка шва
		kernel_div: делитель для ядра шва
	"""

	size = (512, 512)
	gray = random.randint(10, 40)
	sharpness = random.randint(15, 50)
	start_point = 0
	end_point = 0

	# Создаем новое изображение в режиме L (черно-белое)
	image = Image.new('L', size)
	pixels = image.load()

	# Добавляем шум для создания зернистости
	for x in range(size[0]):
		for y in range(size[1]):
			# Генерируем случайное отклонение
			noise = random.randint(-sharpness, sharpness)
			# Применяем шум к базовому цвету

			pixels[x, y] = min(255, gray + noise)

	# Применяем размытие для контроля резкости
	image = image.filter(ImageFilter.GaussianBlur(radius=0.8))

	draw = ImageDraw.Draw(image)
	width, height = image.size

	# Вычисляем шаг между чешуйками с учетом перекрытия
	step = int(pattern_size * (1 - overlap_ratio))
	if step < 1:
		step = 1

	# Определяем направление шва
	direction = random.choice(['horizontal', 'vertical', 'diagonal', 'circle'])
	centers = []
	seam_fill = random.randint(140, 180)

	if direction == 'horizontal':
		start_y = random.randint(0, height)
		start_point = (0, start_y)
		end_point = (width, start_y)

		for x in range(-pattern_size, width + pattern_size, step):
			y_offset = random.randint(-1, 1) * random.random()
			y = start_y + y_offset
			centers.append((x, y))
			# Исправлено: используем квадратный bounding box для круга
			radius = pattern_size / 2
			draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=seam_fill,
						 outline=seam_fill - 20)

	elif direction == 'vertical':
		start_x = random.randint(0, width)
		start_point = (start_x, 0)
		end_point = (start_x, height)

		for y in range(-pattern_size, height + pattern_size, step):
			x_offset = random.randint(-1, 1) * random.random()
			x = start_x + x_offset
			centers.append((x, y))
			# Исправлено: используем квадратный bounding box для круга
			radius = pattern_size / 2
			draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill=seam_fill,
						 outline=seam_fill - 20)

	elif direction == 'circle':
		# Параметры эллипса (первая дуга)
		radius = random.randint(50, 130)
		lin_dist = 50

		# Вычисляем минимальный угол дуги
		try:
			central_angle_rad = math.acos((2 * radius ** 2 - lin_dist ** 2) / (2 * radius ** 2))
		except:
			central_angle_rad = 0
		central_angle_deg = math.degrees(central_angle_rad)
		central_angle = max(random.randint(0, 180), central_angle_deg)

		# Центр первой окружности
		central_pos1 = (random.randint(pattern_size + radius, width - (pattern_size + radius)),
						random.randint(pattern_size + radius, height - (pattern_size + radius)))

		# Случайные начальный и конечный углы для первой дуги
		start_angle1 = random.uniform(0, 180)
		arc_length1 = random.uniform(central_angle, 180)
		end_angle1 = start_angle1 + arc_length1
		normal = start_angle1 + arc_length1 / 2

		# Рисуем первую дугу
		angular_step_deg = math.degrees(step / radius)
		n1 = int(arc_length1 / angular_step_deg) + 1

		start_point1 = (central_pos1[0] + radius * math.cos(math.radians(start_angle1)),
						central_pos1[1] + radius * math.sin(math.radians(start_angle1)))
		end_point1 = (central_pos1[0] + radius * math.cos(math.radians(end_angle1)),
					  central_pos1[1] + radius * math.sin(math.radians(end_angle1)))


		start_point = start_point1
		end_point = end_point1

		# Вычисляем вектор между начальной и конечной точками (большая ось)
		dx = abs(end_point[0] - start_point[0] + pattern_size)
		dy = abs(end_point[1] - start_point[1] + pattern_size)

		centers = []
		for i in range(n1):
			angle = start_angle1 + i * (arc_length1 / n1)
			angle_rad = math.radians(angle)
			x = central_pos1[0] + radius * math.cos(angle_rad)
			y = central_pos1[1] + radius * math.sin(angle_rad)

			x_offset = random.uniform(-1, 1)
			y_offset = random.uniform(-1, 1)
			centers.append((x + x_offset, y + y_offset))

			radius_pattern = pattern_size / 2
			draw.ellipse([x - radius_pattern + x_offset, y - radius_pattern + y_offset,
						  x + radius_pattern + x_offset, y + radius_pattern + y_offset],
						 fill=seam_fill, outline=seam_fill - 20)

		# Находим точки начала и конца первой дуги
		start_point1 = (central_pos1[0] + radius * math.cos(math.radians(start_angle1)),
						central_pos1[1] + radius * math.sin(math.radians(start_angle1)))
		end_point1 = (central_pos1[0] + radius * math.cos(math.radians(end_angle1)),
					  central_pos1[1] + radius * math.sin(math.radians(end_angle1)))


		# Создаем вторую дугу (зеркальную)
		mid_point = ((start_point1[0] + end_point1[0]) / 2, (start_point1[1] + end_point1[1]) / 2)
		dx = central_pos1[0] - mid_point[0]
		dy = central_pos1[1] - mid_point[1]
		central_pos2 = (mid_point[0] - dx, mid_point[1] - dy)

		# Углы для второй дуги
		start_angle2 = math.degrees(math.atan2(start_point1[1] - central_pos2[1], start_point1[0] - central_pos2[0]))
		end_angle2 = math.degrees(math.atan2(end_point1[1] - central_pos2[1], end_point1[0] - central_pos2[0]))

		# Корректируем углы для непрерывности дуги
		if abs(end_angle2 - start_angle2) > 180:
			if end_angle2 > start_angle2:
				end_angle2 -= 360
			else:
				start_angle2 -= 360

		# Рисуем вторую дугу
		arc_length2 = end_angle2 - start_angle2
		n2 = int(abs(arc_length2) / angular_step_deg) + 1

		for i in range(n2, 0, -1):
			angle = start_angle2 + i * (arc_length2 / n2)
			angle_rad = math.radians(angle)
			x = central_pos2[0] + radius * math.cos(angle_rad)
			y = central_pos2[1] + radius * math.sin(angle_rad)

			x_offset = random.uniform(-1, 1)
			y_offset = random.uniform(-1, 1)
			centers.append((x + x_offset, y + y_offset))

			radius_pattern = pattern_size / 2

		middle_point = centers[len(centers) // 4]

		dx = math.sqrt((central_pos1[0] - middle_point[0])**2 + (central_pos1[1] - middle_point[1])**2)

		parallel_line_1 = create_parallel_line(central_pos1[0], central_pos1[1], middle_point[0], middle_point[1], dx/1.1, 2000)
		parallel_line_2 = create_parallel_line(central_pos1[0], central_pos1[1], middle_point[0], middle_point[1], -dx/1.1, 2000)

		width = random.randint(3,9)
		draw.line((parallel_line_1[0], parallel_line_1[1], parallel_line_1[2], parallel_line_1[3]), 150, width)
		draw.line((parallel_line_2[0], parallel_line_2[1], parallel_line_2[2], parallel_line_2[3]), 150, width)


		for i in range(n1*2):
			draw.ellipse([centers[i][0] - radius_pattern, centers[i][1] - radius_pattern,
						  centers[i][0] + radius_pattern, centers[i][1] + radius_pattern],
						 fill=seam_fill, outline=seam_fill - 20)

	elif direction == 'diagonal':
		edge = random.choice(['top', 'bottom', 'left', 'right'])

		if edge == 'top':
			start_x = random.randint(0, width)
			start_y = 0
			base_angle = 90
		elif edge == 'bottom':
			start_x = random.randint(0, width)
			start_y = height
			base_angle = 270
		elif edge == 'left':
			start_x = 0
			start_y = random.randint(0, height)
			base_angle = 0
		else:
			start_x = width
			start_y = random.randint(0, height)
			base_angle = 180

		angle_deviation = random.uniform(-70, 70)
		angle_degrees = base_angle + angle_deviation
		angle_degrees = angle_degrees % 360
		angle = math.radians(angle_degrees)

		if edge == 'top':
			t = (height - start_y) / math.sin(angle)
			end_x = start_x + t * math.cos(angle)
			end_y = height
		elif edge == 'bottom':
			t = (0 - start_y) / math.sin(angle)
			end_x = start_x + t * math.cos(angle)
			end_y = 0
		elif edge == 'left':
			t = (width - start_x) / math.cos(angle)
			end_x = width
			end_y = start_y + t * math.sin(angle)
		else:
			t = (0 - start_x) / math.cos(angle)
			end_x = 0
			end_y = start_y + t * math.sin(angle)

		end_x = max(0, min(width, end_x))
		end_y = max(0, min(height, end_y))

		start_point = (start_x, start_y)
		end_point = (end_x, end_y)
		length = math.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)

		steps = int(length / step)
		for i in range(steps + 1):
			t = i / steps if steps > 0 else 0
			x = start_x + t * (end_x - start_x)
			y = start_y + t * (end_y - start_y)

			if x < -pattern_size or x > width + pattern_size or y < -pattern_size or y > height + pattern_size:
				continue

			offset_x = random.randint(-1, 1) * random.random()
			offset_y = random.randint(-1, 1) * random.random()
			centers.append((x + offset_x, y + offset_y))
			# Исправлено: используем квадратный bounding box для круга
			radius = pattern_size / 2
			draw.ellipse([x - radius + offset_x, y - radius + offset_y, x + radius + offset_x, y + radius + offset_y],
						 fill=seam_fill, outline=seam_fill - 20)

	white_centers = []
	for i in range(len(centers) - 1):
		white_centers.append(centers[i])
		# Не добавляем промежуточную точку между последней точкой первой дуги и первой точкой второй дуги
		if not direction == 'circle' or (i != n1 - 1):
			white_centers.append(((centers[i][0] + centers[i + 1][0]) / 2,
								  (centers[i][1] + centers[i + 1][1]) / 2))

	kernel_radius = pattern_size / (kernel_div * 2)
	for center_x, center_y in white_centers:
		draw.ellipse([center_x - kernel_radius, center_y - kernel_radius,
					  center_x + kernel_radius, center_y + kernel_radius],
					 fill=seam_fill + 30)

	return image, draw, centers, direction, start_point, end_point, pattern_size


