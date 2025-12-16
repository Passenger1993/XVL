from PIL import Image, ImageDraw, ImageFilter
import random, datetime
import math
from generate_seam import draw_a_seam
import statistics
from functools import reduce
from collections import deque
import json
from tqdm import tqdm

def flatten(lst):
    return reduce(lambda acc, x: acc + (flatten(x) if isinstance(x, list) else [x]), lst, [])

def distance(p1, p2):
    """Вычисляет евклидово расстояние между двумя точками"""
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def find_clusters(points, max_distance=70):
    """Находит кластеры точек с использованием BFS"""
    clusters = []
    visited = set()

    for i, point in enumerate(points):
        if i in visited:
            continue

        # Создаем новый кластер
        cluster = []
        queue = deque([i])
        visited.add(i)

        while queue:
            current_idx = queue.popleft()
            current_point = points[current_idx]
            cluster.append(current_point)

            # Ищем соседние точки
            for j, other_point in enumerate(points):
                if j not in visited and distance(current_point, other_point) <= max_distance:
                    visited.add(j)
                    queue.append(j)

        clusters.append(cluster)

    return clusters

def calculate_cluster_info(cluster):
    """Вычисляет центр, ширину и высоту для кластера"""
    if not cluster:
        return None

    # Находим границы кластера
    x_coords = [p[0] for p in cluster]
    y_coords = [p[1] for p in cluster]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    # Центр кластера
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Ширина и высота прямоугольника
    width = max_x - min_x
    height = max_y - min_y

    return {
        'center': (center_x, center_y),
        'width': width,
        'height': height,
        'min_x': min_x,
        'max_x': max_x,
        'min_y': min_y,
        'max_y': max_y,
        'points_count': len(cluster)
    }

def cluster_points(points, max_distance=70):
    """Основная функция для кластеризации точек"""
    # Находим кластеры
    clusters = find_clusters(points, max_distance)

    # Вычисляем информацию для каждого кластера
    cluster_info = []
    for i, cluster in enumerate(clusters):
        info = calculate_cluster_info(cluster)
        if info:
            info['cluster_id'] = i
            info['points'] = cluster
            cluster_info.append(info)

    return cluster_info

def make_pore(num_pores, max_pore_area=70):
    """
    Добавляет поры на изображение

    Args:
        num_pores: количество пор для добавления
        max_pore_area: максимальная площадь одной поры в пикселях

    Returns:
        Изображение с порами
    """
    pore_color = random.randint(20,70)
    result, draw, centers, direction, start_point, end_point, pattern_size, seam_area = draw_a_seam()
    pore_centers = [] # список центров пор
    sorted_pores = [] # 2-мерный массив пор по кластерам
    pore_centers.append(random.choice(centers))

    if num_pores != 1:
        for i in range(5):
            if random.random()>0.7:
                pore_centers.append(random.choice(centers))

    for center in pore_centers:
        sorted_pores.append([])
        if num_pores == 1:
            pore_range = range(1)
        else:
            pore_range = range(random.randint(2,30))
        for pore in pore_range:
            # Ограничиваем координаты центра поры пределами шва
            # Используем pattern_size как максимальное отклонение от центра шва
            max_deviation = pattern_size // 2  # Уменьшаем область для гарантии нахождения внутри шва
            
            center_x = center[0] + random.randint(-max_deviation, max_deviation)
            center_y = center[1] + random.randint(-max_deviation, max_deviation)
            
            # Дополнительная проверка, чтобы поры не выходили за границы изображения
            center_x = max(0, min(center_x, result.width - 1))
            center_y = max(0, min(center_y, result.height - 1))

            sorted_pores[-1].append((center_x, center_y))

            # Ограничиваем максимальный размер поры
            max_radius = int(math.sqrt(max_pore_area / math.pi))
            radius = random.randint(2, max_radius)
            
            # Дополнительная проверка: ограничиваем радиус, чтобы пора не выходила за пределы шва
            max_allowed_radius = min(max_radius, max_deviation)
            radius = min(radius, max_allowed_radius)

            # Случайное искажение формы (эллипс или более сложная форма)
            shape_type = random.choice(['circle', 'irregular'])

            if shape_type == 'circle':
                # Рисуем эллипс
                draw.circle((center_x, center_y), radius, fill=pore_color)  # Черный цвет

            else:
                # Создаем неправильную форму
                points = []
                num_points = random.randint(10, 25)

                # Генерируем точки вокруг центра с ограничением радиуса
                for i in range(num_points):
                    angle = 2 * math.pi * i / num_points
                    # Добавляем случайное отклонение к радиусу, но не больше допустимого
                    r_x = radius * (0.7 + 0.6 * random.random())
                    r_y = radius * (0.7 + 0.6 * random.random())
                    
                    # Ограничиваем координаты точек, чтобы они не выходили за пределы шва
                    x = center_x + r_x * math.cos(angle)
                    y = center_y + r_y * math.sin(angle)
                    
                    # Проверяем, что точка не слишком далеко от центра шва
                    if distance((x, y), center) <= pattern_size:
                        points.append((x, y))
                    else:
                        # Если точка выходит за пределы, используем ближайшую допустимую точку
                        angle_to_center = math.atan2(y - center[1], x - center[0])
                        x = center[0] + (pattern_size * 0.9) * math.cos(angle_to_center)
                        y = center[1] + (pattern_size * 0.9) * math.sin(angle_to_center)
                        points.append((x, y))

                # Рисуем многоугольник только если есть достаточно точек
                if len(points) >= 3:
                    draw.polygon(points, fill=pore_color)

    dist_treshold = 30
    clusters = cluster_points(flatten(sorted_pores), dist_treshold)

    data = {}
    for i,cluster in enumerate(clusters):
        data[f"PC{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"+str(cluster['cluster_id'])] = [cluster['min_x']-pattern_size,cluster['min_y']-pattern_size,cluster['max_x']+pattern_size,cluster['max_y']+pattern_size]
    return result, data

def generate_dataset(directory, num_images=100, num_pores = 1):
    summary = []
    progress_bar = tqdm(total=num_images)
    path = "C:/Users/Алексей/Documents/XVL_2025_set"
    if num_pores == 1:
        text = "Генерация одиночных пор"
    else:
        text = "Генерация скоплений пор"
    for i in range(num_images):
        # Создаем зернистое изображение
        pore_image, data = make_pore(num_pores, max_pore_area=250)
        pore_image.save(f"{path}/{directory}/PC{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
        summary.append(data)
        progress_bar.set_description(f"{text} № {i+1}")
        progress_bar.update(1)
    with open(f"{directory}/summary.jsom","w") as file:
        json.dump(summary, file)

def test():
    pores,data = make_pore(10)
    draw = ImageDraw.Draw(pores)
    for key in data.keys():
        draw.rectangle([data[key][0], data[key][1], data[key][2], data[key][3]], outline="red", width=2)
    pores.show()
