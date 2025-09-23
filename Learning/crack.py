import math
import random
from typing import List, Tuple
import datetime
import json, tqdm
from generate_seam import draw_a_seam

def draw_variable_thickness_line(draw, start_point, end_point, mean_thickness):
    """
    Рисует линию с плавно меняющейся толщиной.
    """
    num_segments = 20
    start_thickness = random.uniform(0.8 * mean_thickness, 1.2 * mean_thickness)
    end_thickness = random.uniform(0.8 * mean_thickness, 1.2 * mean_thickness)

    for i in range(num_segments):
        t1 = i / num_segments
        t2 = (i + 1) / num_segments

        seg_start = (
            start_point[0] + t1 * (end_point[0] - start_point[0]),
            start_point[1] + t1 * (end_point[1] - start_point[1])
        )
        seg_end = (
            start_point[0] + t2 * (end_point[0] - start_point[0]),
            start_point[1] + t2 * (end_point[1] - start_point[1])
        )

        t_mid = (t1 + t2) / 2
        thickness = start_thickness + t_mid * (end_thickness - start_thickness)

        draw.line([seg_start, seg_end], fill=0, width=int(round(thickness)))

def draw_tapering_crack_line(draw, start_point, end_point, mean_thickness, is_dead_end=False):
    """
    Рисует линию трещины с плавным сужением к концу (для тупиковых ветвей).
    """
    if is_dead_end:
        num_segments = 30
        length = math.sqrt((end_point[0] - start_point[0])**2 + (end_point[1] - start_point[1])**2)

        for i in range(num_segments):
            t1 = i / num_segments
            t2 = (i + 1) / num_segments

            thickness_factor = 1 - (i / num_segments) * 0.8
            current_thickness = max(1, mean_thickness * thickness_factor)
            current_thickness *= random.uniform(0.9, 1.1)

            seg_start = (
                start_point[0] + t1 * (end_point[0] - start_point[0]),
                start_point[1] + t1 * (end_point[1] - start_point[1])
            )
            seg_end = (
                start_point[0] + t2 * (end_point[0] - start_point[0]),
                start_point[1] + t2 * (end_point[1] - start_point[1])
            )

            draw.line([seg_start, seg_end], fill=0, width=int(round(current_thickness)))
    else:
        draw_variable_thickness_line(draw, start_point, end_point, mean_thickness)

def draw_crack_branch(draw, start_point, angle, thickness, remaining_thicknesses, is_main_branch=False) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Рекурсивная функция для рисования ветви трещины.
    Возвращает координаты ограничивающего прямоугольника (min_x, min_y, max_x, max_y).
    """
    # Инициализируем ограничивающий прямоугольник
    min_x, min_y = start_point
    max_x, max_y = start_point

    # Определяем длину сегмента
    if is_main_branch:
        length = random.randint(40, 80)
    else:
        length = random.randint(20, 60)

    end_x = start_point[0] + length * math.cos(angle)
    end_y = start_point[1] + length * math.sin(angle)
    end_point = (end_x, end_y)

    # Обновляем ограничивающий прямоугольник
    min_x = min(min_x, end_x)
    min_y = min(min_y, end_y)
    max_x = max(max_x, end_x)
    max_y = max(max_y, end_y)

    # Определяем, является ли эта ветвь тупиковой
    is_dead_end = not remaining_thicknesses

    # Рисуем сегмент трещины
    draw_tapering_crack_line(draw, start_point, end_point, thickness, is_dead_end)

    if remaining_thicknesses:
        # Определяем количество ответвлений
        num_branches = random.randint(1, min(3, len(remaining_thicknesses)))
        branches_thicknesses = remaining_thicknesses[:num_branches]
        next_remaining = remaining_thicknesses[num_branches:]

        # Определяем углы для ответвлений
        angles = []
        max_angle = math.pi / 3

        for i in range(num_branches):
            if i == 0 and is_main_branch:
                branch_angle = angle
            else:
                deviation = random.uniform(-max_angle, max_angle)
                branch_angle = angle + deviation
            angles.append(branch_angle)

        # Случайным образом выбираем, какая ветвь будет продолжаться
        continue_index = random.randint(0, num_branches - 1)

        # Рисуем все ответвления и обновляем ограничивающий прямоугольник
        for i in range(num_branches):
            is_continue = (i == continue_index)
            branch_bbox = draw_crack_branch(draw, end_point, angles[i], branches_thicknesses[i],
                                           next_remaining if is_continue else [], is_continue)

            # Обновляем общий ограничивающий прямоугольник
            min_x = min(min_x, branch_bbox[0][0], branch_bbox[1][0])
            min_y = min(min_y, branch_bbox[0][1], branch_bbox[1][1])
            max_x = max(max_x, branch_bbox[0][0], branch_bbox[1][0])
            max_y = max(max_y, branch_bbox[0][1], branch_bbox[1][1])

    return [min_x, min_y, max_x, max_y]

def make_a_crack(divisions: List[float]):
    """
    Рисует трещину на изображении с заданными параметрами ветвления.
    Возвращает изображение и координаты ограничивающего прямоугольника.
    """
    divisions_sorted = sorted(divisions, reverse=True)
    image, draw, centers, direction, start_point, end_point, pattern_size = draw_a_seam()
    width, height = image.size

    # Выбираем случайную стену для начала трещины
    wall = random.choice(['top', 'bottom', 'left', 'right'])
    if wall == 'top':
        start_x = random.randint(0, width)
        start_y = 0
        base_angle = math.pi / 2
    elif wall == 'bottom':
        start_x = random.randint(0, width)
        start_y = height
        base_angle = -math.pi / 2
    elif wall == 'left':
        start_x = 0
        start_y = random.randint(0, height)
        base_angle = 0
    else:  # 'right'
        start_x = width
        start_y = random.randint(0, height)
        base_angle = math.pi

    # Добавляем случайное отклонение к углу
    angle = base_angle + random.uniform(-math.pi/5, math.pi/5)

    # Рисуем трещину и получаем ограничивающий прямоугольник
    bbox = None
    if divisions_sorted:
        bbox = draw_crack_branch(draw, (start_x, start_y), angle, divisions_sorted[0], divisions_sorted[1:], is_main_branch=True)

    return image, bbox

def generate_dataset(directory, num_images=100):
    """
    Генерирует набор изображений с трещинами и соответствующие JSON-файлы с координатами.
    """
    for i in range(num_images):
        progress_bar = tqdm(total=num_images)

        divisions = [random.randint(1,10) for i in range (random.randint(3,15))]

        # Создаем трещину и получаем координаты
        crack_image, bbox = make_a_crack(divisions)

        cords = []

        # Сохраняем изображение
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_filename = f"{directory}/T{i}_{timestamp}.png"
        crack_image.save(image_filename)
        progress_bar.set_description(f"Генерация трещин № {i+1}")
        progress_bar.update(1)

        # Создаем и сохраняем JSON с координатами
        if bbox:
            json_data = {
                f"T{i}_{timestamp}": [
                    bbox[0][0],  # x1
                    bbox[0][1],  # y1
                    bbox[1][0],  # x2
                    bbox[1][1]   # y2
                ]
            }

            cords.append(json_data)

    with open(f"{directory}/summary.json", 'w') as f:
        json.dump(cords, f, indent=2)

