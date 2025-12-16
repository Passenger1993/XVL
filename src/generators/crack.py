import math
import random
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict
import datetime
import json
from tqdm import tqdm
from generate_seam import draw_a_seam
import matplotlib.pyplot as plt
import cv2
import numpy as np

def draw_variable_thickness_line(draw, start_point, end_point, mean_thickness, defect_sharpness=1.0):
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

def get_local_direction(centers, point_index, window_size=5):
    """
    Вычисляет локальное направление шва в заданной точке,
    используя линейную регрессию для близлежащих точек.
    """
    # Определяем диапазон индексов для анализа
    start_idx = max(0, point_index - window_size)
    end_idx = min(len(centers), point_index + window_size + 1)

    # Извлекаем точки для анализа
    points = centers[start_idx:end_idx]

    if len(points) < 2:
        return 0  # Недостаточно точек для определения направления

    # Преобразуем точки в массивы numpy
    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Вычисляем линейную регрессию
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]

    # Вычисляем угол наклона линии
    angle = math.atan(slope)

    return angle

def draw_crack_branch(draw, start_point, angle, thickness, remaining_thicknesses, crack_type, is_main_branch=False, bbox=None):
    """
    Рекурсивная функция для рисования ветви трещины.
    """
    if bbox is None:
        bbox = [float('inf'), float('inf'), float('-inf'), float('-inf')]

    # Определяем длину сегмента
    if is_main_branch:
        length = random.randint(40, 80)
    else:
        length = random.randint(20, 60)

    end_x = start_point[0] + length * math.cos(angle)
    end_y = start_point[1] + length * math.sin(angle)
    end_point = (end_x, end_y)

    # Обновляем ограничивающий прямоугольник
    bbox[0] = min(bbox[0], start_point[0], end_x)
    bbox[1] = min(bbox[1], start_point[1], end_y)
    bbox[2] = max(bbox[2], start_point[0], end_x)
    bbox[3] = max(bbox[3], start_point[1], end_y)

    # Определяем, является ли эта ветвь тупиковой
    is_dead_end = not remaining_thicknesses

    # Рисуем сегмент трещины
    draw_tapering_crack_line(draw, start_point, end_point, thickness, is_dead_end)

    # ДОБАВЛЯЕМ ОБРАБОТКУ ДЛЯ "Single" ТИПА
    if remaining_thicknesses:
        if crack_type == "single":
            # Для одиночной трещины продолжаем в том же направлении с небольшим отклонением
            next_thickness = remaining_thicknesses[0]
            next_remaining = remaining_thicknesses[1:]

            # Небольшое случайное отклонение угла для естественного вида
            deviation = random.uniform(-math.pi/6, math.pi/6)
            next_angle = angle + deviation

            draw_crack_branch(draw, end_point, next_angle, next_thickness,
                             next_remaining, crack_type, is_main_branch=True, bbox=bbox)

        if crack_type == "shattered":
            # Исправлено: гарантируем, что min <= max
            max_branches = min(4, len(remaining_thicknesses))
            min_branches = min(2, max_branches)

            if min_branches <= max_branches:
                num_branches = random.randint(min_branches, max_branches)
            else:
                num_branches = max_branches

            # Равномерно распределяем толщины между ветвями
            branches_thicknesses = [remaining_thicknesses[i % len(remaining_thicknesses)] for i in range(num_branches)]
            next_remaining = remaining_thicknesses[num_branches:] if len(remaining_thicknesses) > num_branches else []

            # Определяем углы для ответвлений
            max_angle = math.pi / 3
            angles = []
            for i in range(num_branches):
                deviation = random.uniform(-max_angle, max_angle)
                branch_angle = angle + deviation
                angles.append(branch_angle)

            # Рисуем все ответвления как равнозначные
            for i in range(num_branches):
                draw_crack_branch(draw, end_point, angles[i], branches_thicknesses[i],
                                 next_remaining, crack_type, is_main_branch=False, bbox=bbox)

        # Для древовидного типа (как раньше)
        elif crack_type == "tree":
            # Исправлено: гарантируем, что min <= max
            max_branches = min(2, len(remaining_thicknesses))
            min_branches = min(1, max_branches)

            if min_branches <= max_branches:
                num_branches = random.randint(min_branches, max_branches)
            else:
                num_branches = max_branches

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
            if num_branches > 0:
                continue_index = random.randint(0, num_branches - 1)

                # Рисуем все ответвления и обновляем ограничивающий прямоугольник
                for i in range(num_branches):
                    is_continue = (i == continue_index)
                    draw_crack_branch(draw, end_point, angles[i], branches_thicknesses[i],
                                     next_remaining if is_continue else [], crack_type,
                                     is_continue, bbox)


    return bbox

def draw_transverse_crack_type(draw, centers, pattern_size):
    """
    Рисует поперечные трещины для типа "transverse"
    Возвращает список bounding box'ов для каждой трещины
    """
    if not centers or len(centers) < 10:
        return []

    # Генерируем случайное количество трещин (от 1 до 4)
    num_cracks = random.randint(1, 4)

    cracks_data = []

    for i in range(num_cracks):
        # Выбираем случайную точку на шве (избегаем краев)
        point_index = random.randint(5, len(centers) - 6)
        center_point = centers[point_index]

        # Вычисляем локальное направление шва в этой точке
        seam_angle = get_local_direction(centers, point_index)

        # Угол поперечной трещины перпендикулярен направлению шва
        transverse_angle = seam_angle + math.pi/2

        # Случайная толщина (1-2 пикселя)
        thickness = random.randint(1, 2)

        # Длина поперечной трещины равна ширине шва (pattern_size)
        crack_length = pattern_size

        # Вычисляем начальную и конечную точки поперечной трещины
        # Трещина перпендикулярна направлению шва
        half_length = crack_length / 2
        start_point = (
            center_point[0] - half_length * math.cos(transverse_angle),
            center_point[1] - half_length * math.sin(transverse_angle)
        )
        end_point = (
            center_point[0] + half_length * math.cos(transverse_angle),
            center_point[1] + half_length * math.sin(transverse_angle)
        )

        # Рисуем поперечную трещину
        draw.line([start_point, end_point], fill=0, width=thickness)

        # Создаем bounding box для этой трещины с отступами
        min_x = min(start_point[0], end_point[0]) - pattern_size
        min_y = min(start_point[1], end_point[1]) - pattern_size
        max_x = max(start_point[0], end_point[0]) + pattern_size
        max_y = max(start_point[1], end_point[1]) + pattern_size

        # Сохраняем данные о трещине
        cracks_data.append({
            'crack_id': i,
            'min_x': min_x,
            'min_y': min_y,
            'max_x': max_x,
            'max_y': max_y
        })

    return cracks_data

def make_a_crack(crack_type=random.choice(["single", "tree", "shattered", "transverse"])):
    """
    Рисует трещину на изображении с заданными параметрами ветвления.
    """
    # ИСПРАВЛЕННАЯ генерация толщин для разных типов трещин
    if crack_type == "single":
        # Для одиночной трещины - больше сегментов для большей длины
        num_segments = random.randint(4, 8)  # Увеличиваем количество сегментов
        divisions = [random.randint(3, 6) for i in range(num_segments)]
    elif crack_type == "shattered":
        divisions = [random.randint(2, 5) for i in range(random.randint(4, 5))]
    elif crack_type == "transverse":
        # Для поперечных трещин толщина не используется в том же смысле
        divisions = [random.randint(1, 3) for i in range(random.randint(1, 3))]
    else:  # tree
        divisions = [random.randint(1, 5) for i in range(random.randint(3, 5))]

    divisions_sorted = sorted(divisions, reverse=True)

    # Получаем изображение шва
    image, draw, centers, direction, start_point, end_point, pattern_size, seam_area = draw_a_seam()
    width, height = image.size

    # Создаем словарь для данных
    data = {}

    # Рисуем трещину в зависимости от типа
    if crack_type == "transverse":
        # Для поперечных трещин используем специальную функцию
        cracks_data = draw_transverse_crack_type(draw, centers, pattern_size)

        # Каждая поперечная трещина - отдельный дефект
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        for i, crack in enumerate(cracks_data):
            defect_id = f"C{i}_{timestamp}{crack['crack_id']}"
            data[defect_id] = [
                max(0, crack['min_x']),
                max(0, crack['min_y']),
                min(width, crack['max_x']),
                min(height, crack['max_y'])
            ]

    else:
        # Для остальных типов трещин - одна трещина в одном bounding box

        # Для продольных трещин выбираем точку на шве
        if centers and len(centers) > 10:
            # Выбираем случайную точку на шве (избегаем краев)
            point_index = random.randint(5, len(centers) - 6)
            start_x, start_y = centers[point_index]

            # Вычисляем локальное направление шва в этой точке
            base_angle = get_local_direction(centers, point_index)
        else:
            # Резервный вариант - случайная точка на изображении и направление
            start_x = random.randint(int(0.1 * width), int(0.9 * width))
            start_y = random.randint(int(0.1 * height), int(0.9 * height))
            base_angle = random.uniform(0, 2 * math.pi)

        bbox_coords = None
        if divisions_sorted:
            # Первое направление
            bbox1 = draw_crack_branch(draw, (start_x, start_y), base_angle, divisions_sorted[0],
                                     divisions_sorted[1:], crack_type, is_main_branch=True)

            # Второе направление (противоположное)
            bbox2 = draw_crack_branch(draw, (start_x, start_y), base_angle + math.pi, divisions_sorted[0],
                                     divisions_sorted[1:], crack_type, is_main_branch=True)

            # Объединяем ограничивающие прямоугольники
            combined_bbox = [
                min(bbox1[0], bbox2[0]),
                min(bbox1[1], bbox2[1]),
                max(bbox1[2], bbox2[2]),
                max(bbox1[3], bbox2[3])
            ]

            # Обеспечиваем, что координаты не выходят за границы изображения
            combined_bbox[0] = max(0, combined_bbox[0])
            combined_bbox[1] = max(0, combined_bbox[1])
            combined_bbox[2] = min(width, combined_bbox[2])
            combined_bbox[3] = min(height, combined_bbox[3])

            # Проверяем, что прямоугольник имеет положительную площадь
            if combined_bbox[2] > combined_bbox[0] and combined_bbox[3] > combined_bbox[1]:
                bbox_coords = combined_bbox

        # Создаем запись для одиночной трещины
        if bbox_coords:
            defect_id = f"C{crack_type}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            data[defect_id] = bbox_coords

    return image, data

def test():
    for i in range(4):
        # Тестируем все четыре типа трещин
        crack_types = ["single", "tree", "shattered", "transverse"]
        crack = make_a_crack(crack_types[i])
        draw = ImageDraw.Draw(crack[0])

        # Рисуем bounding boxes
        for defect_id, bbox in crack[1].items():
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=2)

        crack[0].show()

def generate_dataset(directory, num_images=100):
    """
    Генерирует набор изображений с трещинами и соответствующие JSON-файлы с координатами.
    """
    progress_bar = tqdm(total=num_images)
    all_coords = {}
    path = "C:/Users/Алексей/Documents/XVL_2025_set"

    for i in range(num_images):
        progress_bar.set_description(f"Генерация трещин № {i+1}")
        progress_bar.update(1)

        # Случайным образом выбираем тип трещины
        crack_type = random.choice(["single", "tree", "shattered", "transverse"])

        # Создаем трещину и получаем координаты
        crack_image, bbox_dict = make_a_crack(crack_type)

        # Сохраняем изображение
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_filename = f"{path}/{directory}/T{i}_{crack_type}_{timestamp}.png"
        crack_image.save(image_filename)

        # Добавляем координаты в общий словарь
        for defect_id, coords in bbox_dict.items():
            all_coords[defect_id] = coords

    # Сохраняем все координаты в один JSON файл
    with open(f"{directory}/summary.json", 'w') as f:
        json.dump(all_coords, f, indent=2)

    progress_bar.close()

