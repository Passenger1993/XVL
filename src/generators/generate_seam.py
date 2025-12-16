from PIL import Image, ImageDraw, ImageFilter, ImageChops, ImageFont
import random, math
import numpy as np

def apply_lighting_gradient(image, angle_degrees, intensity):
    """
    Применяет градиент освещения к изображению.

    Args:
        image: изображение в режиме 'L'
        angle_degrees: угол падения света (0-360 градусов)
        intensity: интенсивность освещения (0-1, где 0 - нет эффекта, 1 - максимальный эффект)

    Returns:
        Изображение с примененным градиентом освещения
    """
    if intensity <= 0:
        return image

    # Преобразуем угол в радианы
    angle_rad = math.radians(angle_degrees)

    # Создаем градиентное изображение
    width, height = image.size
    gradient = Image.new('L', (width, height))
    gradient_draw = ImageDraw.Draw(gradient)

    # Вычисляем вектор направления света
    dx = math.cos(angle_rad)
    dy = math.sin(angle_rad)

    # Нормализуем вектор
    length = math.sqrt(dx*dx + dy*dy)
    if length > 0:
        dx /= length
        dy /= length

    # Центр изображения
    center_x = width / 2
    center_y = height / 2

    # Максимальное расстояние от центра до угла
    max_distance = math.sqrt(center_x**2 + center_y**2)

    # Создаем градиент
    for x in range(width):
        for y in range(height):
            # Вектор от центра к точке
            vx = x - center_x
            vy = y - center_y

            # Проекция на направление света
            projection = vx * dx + vy * dy

            # Нормализуем проекцию
            normalized_projection = projection / max_distance

            # Вычисляем яркость (от 0 до 255)
            # Чем больше проекция, тем ярче точка
            brightness = 128 + int(127 * normalized_projection * intensity)

            # Ограничиваем диапазон
            brightness = max(0, min(255, brightness))

            # Устанавливаем пиксель
            gradient.putpixel((x, y), brightness)

    # Применяем градиент к изображению с помощью screen (осветление)
    result = ImageChops.screen(image, gradient)

    return result

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

def generate_random_text(length):
    """Генерирует случайный текст заданной длины"""
    # Символы для генерации текста
    cyrillic = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя'
    latin = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    digits = '0123456789'

    # Комбинируем все символы
    all_chars = cyrillic + latin + digits

    # Генерируем текст случайной длины
    text_length = random.randint(1, length)
    return ''.join(random.choice(all_chars) for _ in range(text_length))

def calculate_seam_area(centers, direction, pattern_size, start_point=None, end_point=None, parallel_lines=None):
    """
    Вычисляет ограничивающий прямоугольник (bounding box) для шва

    Args:
        centers: центры чешуек шва
        direction: направление шва
        pattern_size: размер узора (чешуйки)
        start_point: начальная точка шва
        end_point: конечная точка шва
        parallel_lines: для кругового шва - кортеж с двумя параллельными линиями

    Returns:
        Кортеж (x_min, y_min, x_max, y_max) - ограничивающий прямоугольник шва
    """
    if not centers:
        return (0, 0, 0, 0)

    # Инициализируем минимальные и максимальные значения
    all_x = []
    all_y = []

    # Добавляем центры чешуек с учетом радиуса чешуйки
    radius = pattern_size / 2
    for center in centers:
        all_x.append(center[0] - radius)
        all_x.append(center[0] + radius)
        all_y.append(center[1] - radius)
        all_y.append(center[1] + radius)

    # Для линейных швов добавляем начальную и конечную точки
    if direction in ['horizontal', 'vertical', 'diagonal'] and start_point and end_point:
        all_x.append(start_point[0])
        all_x.append(end_point[0])
        all_y.append(start_point[1])
        all_y.append(end_point[1])

    # Для кругового шва добавляем точки параллельных линий
    if direction == 'circle' and parallel_lines:
        line1, line2 = parallel_lines
        all_x.append(line1[0])
        all_x.append(line1[2])
        all_x.append(line2[0])
        all_x.append(line2[2])
        all_y.append(line1[1])
        all_y.append(line1[3])
        all_y.append(line2[1])
        all_y.append(line2[3])

    # Вычисляем границы
    x_min = min(all_x)
    x_max = max(all_x)
    y_min = min(all_y)
    y_max = max(all_y)

    # Добавляем отступ (5 мм ≈ 20 пикселей при стандартном разрешении)
    margin = 20
    x_min = max(0, x_min - margin)
    y_min = max(0, y_min - margin)
    x_max = min(512, x_max + margin)
    y_max = min(512, y_max + margin)

    return (x_min, y_min, x_max, y_max)

def is_point_near_seam_area(x, y, seam_area, min_distance=50):
    """
    Проверяет, находится ли точка слишком близко к области шва

    Args:
        x, y: координаты точки
        seam_area: область шва (x_min, y_min, x_max, y_max)
        min_distance: минимальное расстояние до шва в пикселях
    """
    x_min, y_min, x_max, y_max = seam_area

    # Если точка внутри области шва (с учетом отступа)
    if x_min - min_distance <= x <= x_max + min_distance and y_min - min_distance <= y <= y_max + min_distance:
        return True

    # Проверяем расстояние до каждого угла прямоугольника
    corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]

    for cx, cy in corners:
        distance = math.sqrt((x - cx)**2 + (y - cy)**2)
        if distance < min_distance:
            return True

    # Проверяем расстояние до каждой стороны прямоугольника
    # Слева
    if x_min - min_distance <= x <= x_min and y_min <= y <= y_max:
        return True
    # Справа
    if x_max <= x <= x_max + min_distance and y_min <= y <= y_max:
        return True
    # Сверху
    if y_min - min_distance <= y <= y_min and x_min <= x <= x_max:
        return True
    # Снизу
    if y_max <= y <= y_max + min_distance and x_min <= x <= x_max:
        return True

    return False

def get_random_font(font_size):
    """
    Пытается получить случайный шрифт из списка доступных.
    Возвращает шрифт или стандартный, если ни один из указанных не найден.
    """
    # Список шрифтов для случайного выбора
    font_options = [
        ("arialbd.ttf", "Жирный Arial"),
        ("arial.ttf", "Обычный Arial"),
        ("DejaVuSans-Bold.ttf", "Жирный DejaVu Sans"),
        ("DejaVuSans.ttf", "Обычный DejaVu Sans"),
        ("timesbd.ttf", "Жирный Times New Roman"),
        ("times.ttf", "Обычный Times New Roman"),
        ("courbd.ttf", "Жирный Courier New"),
        ("cour.ttf", "Обычный Courier New"),
        ("verdanab.ttf", "Жирный Verdana"),
        ("verdana.ttf", "Обычный Verdana"),
        ("tahoma.ttf", "Tahoma"),
        ("tahomabd.ttf", "Жирный Tahoma"),
        ("calibri.ttf", "Calibri"),
        ("calibrib.ttf", "Жирный Calibri"),
        ("consola.ttf", "Consolas"),
        ("consolab.ttf", "Жирный Consolas"),
    ]

    # Перемешиваем список шрифтов
    random.shuffle(font_options)

    # Пытаемся загрузить каждый шрифт по очереди
    for font_file, font_name in font_options:
        try:
            font = ImageFont.truetype(font_file, font_size)
            return font
        except:
            continue

    # Если ни один из шрифтов не загрузился, используем стандартный
    return ImageFont.load_default()

def draw_random_text(draw, seam_area, pattern_size, gray_color, size=(512, 512)):
    """
    Рисует текст в случайных местах, но не ближе 50 пикселей к шву.
    Гарантирует, что будет нарисован хотя бы 1 текст.

    Args:
        draw: объект ImageDraw для рисования
        seam_area: область шва (x_min, y_min, x_max, y_max)
        pattern_size: размер узора для расчета минимального расстояния между текстами
        gray_color: базовый цвет для расчета цвета текста
        size: размер изображения
    """
    # Генерируем случайное количество текстовых элементов (минимум 3, максимум 8)
    num_texts = random.randint(3, 8)

    # Минимальное расстояние между текстовыми элементами
    min_text_distance = pattern_size * 3

    # Минимальное расстояние до шва
    min_seam_distance = 50

    # Определяем запасные позиции в углах изображения
    backup_positions = [
        (50, 50),
        (size[0] - 50, 50),
        (50, size[1] - 50),
        (size[0] - 50, size[1] - 50)
    ]

    # Подготовка позиций для текста
    text_positions = []
    attempts = 0
    max_attempts = num_texts * 50

    # Сначала попробуем найти случайные позиции
    while len(text_positions) < num_texts and attempts < max_attempts:
        # Генерируем случайную позицию
        x = random.randint(30, size[0] - 30)
        y = random.randint(30, size[1] - 30)

        # Проверяем, что точка не слишком близко к шву
        if is_point_near_seam_area(x, y, seam_area, min_seam_distance):
            attempts += 1
            continue

        # Проверяем расстояние до других текстовых позиций
        too_close_to_other_text = False
        for pos in text_positions:
            dist = math.sqrt((x - pos[0])**2 + (y - pos[1])**2)
            if dist < min_text_distance:
                too_close_to_other_text = True
                break

        if too_close_to_other_text:
            attempts += 1
            continue

        # Если все проверки пройдены, добавляем позицию
        text_positions.append((x, y))
        attempts = 0

    # Если не удалось разместить достаточное количество текста, используем запасные позиции
    if len(text_positions) < 3:
        for pos in backup_positions:
            if not is_point_near_seam_area(pos[0], pos[1], seam_area, min_seam_distance):
                text_positions.append(pos)
                if len(text_positions) >= 3:
                    break

    # Если все еще нет позиций, добавляем любые позиции
    if not text_positions:
        text_positions = backup_positions[:3]

    # Генерируем текстовые элементы
    text_options = []
    for _ in range(max(10, len(text_positions))):
        text_options.append(generate_random_text(5))

    # Увеличиваем размер шрифта
    base_font_size = random.randint(18, 24)
    font_size_variation = random.randint(-2, 2)
    font_size = max(16, base_font_size + font_size_variation)

    # Получаем случайный шрифт
    font = get_random_font(font_size)

    # Делаем текст светлее - от светло-серого до почти белого
    text_color_min = min(255, gray_color + 50)
    text_color_max = min(255, gray_color + 100)
    text_color = random.randint(text_color_min, text_color_max)

    # Рисуем текст в вычисленных позициях
    for i, pos in enumerate(text_positions):
        if i >= len(text_options):
            break

        text = text_options[i % len(text_options)]

        # Для лучшей читаемости добавляем небольшой черный контур
        outline_color = max(0, text_color - 80)

        # Рисуем контур (смещая на 1 пиксель в разные стороны)
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                draw.text((pos[0] + dx, pos[1] + dy), text, fill=outline_color, font=font)

        # Рисуем основной текст
        draw.text(pos, text, fill=text_color, font=font)

def is_rect_intersect(rect1, rect2):
    """Проверяет, пересекаются ли два прямоугольника"""
    x1_min, y1_min, x1_max, y1_max = rect1
    x2_min, y2_min, x2_max, y2_max = rect2

    # Проверяем пересечение по осям
    if x1_max < x2_min or x1_min > x2_max:
        return False
    if y1_max < y2_min or y1_min > y2_max:
        return False
    return True

def normalize_rectangle_coords(x0, y0, x1, y1):
    """
    Нормализует координаты прямоугольника, гарантируя что x0 <= x1 и y0 <= y1
    """
    if x0 > x1:
        x0, x1 = x1, x0
    if y0 > y1:
        y0, y1 = y1, y0
    return (x0, y0, x1, y1)

def draw_sensitivity_stripe(draw, seam_area, image_size=(512, 512)):
    """
    Рисует светлую полоску (эталон чувствительности) за пределами шва.
    Гарантирует, что полоска будет нарисована на каждом изображении.

    Args:
        draw: объект ImageDraw для рисования
        seam_area: область шва (x_min, y_min, x_max, y_max)
        image_size: размер изображения
    """
    # Параметры полоски
    stripe_width = random.randint(30, 52)  # ширина полоски
    stripe_length = random.randint(135, 165)  # длина полоски

    # Случайно выбираем ориентацию полоски
    orientation = random.choice(['horizontal', 'vertical'])

    # Определяем возможные позиции для полоски (4 угла)
    possible_positions = []

    if orientation == 'horizontal':
        # Горизонтальные позиции в углах
        possible_positions.append((20, 20))
        possible_positions.append((20, image_size[1] - stripe_width - 20))
        possible_positions.append((image_size[0] - stripe_length - 20, 20))
        possible_positions.append((image_size[0] - stripe_length - 20, image_size[1] - stripe_width - 20))
    else:
        # Вертикальные позиции в углах
        possible_positions.append((20, 20))
        possible_positions.append((image_size[0] - stripe_width - 20, 20))
        possible_positions.append((20, image_size[1] - stripe_length - 20))
        possible_positions.append((image_size[0] - stripe_width - 20, image_size[1] - stripe_length - 20))

    # Перемешиваем позиции
    random.shuffle(possible_positions)

    # Пытаемся найти подходящее место
    for pos in possible_positions:
        if orientation == 'horizontal':
            # Нормализуем координаты прямоугольника
            x0, y0 = pos
            x1 = x0 + stripe_length
            y1 = y0 + stripe_width
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
            stripe_rect = (x0, y0, x1, y1)
        else:
            # Нормализуем координаты прямоугольника
            x0, y0 = pos
            x1 = x0 + stripe_width
            y1 = y0 + stripe_length
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
            stripe_rect = (x0, y0, x1, y1)

        # Проверяем, что полоска не пересекается с областью шва
        if not is_rect_intersect(stripe_rect, seam_area):
            # Нашли подходящее место, рисуем полоску
            stripe_color = random.randint(200, 230)
            draw.rectangle(stripe_rect, fill=stripe_color)

            # Рисуем элементы полоски (круги и зебру)
            draw_stripe_elements(draw, pos, orientation, stripe_width, stripe_length)

            return

    # Если не нашли подходящего места, рисуем в первом углу в любом случае
    pos = possible_positions[0]
    if orientation == 'horizontal':
        # Нормализуем координаты прямоугольника
        x0, y0 = pos
        x1 = x0 + stripe_length
        y1 = y0 + stripe_width
        x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
        stripe_rect = (x0, y0, x1, y1)
    else:
        # Нормализуем координаты прямоугольника
        x0, y0 = pos
        x1 = x0 + stripe_width
        y1 = y0 + stripe_length
        x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
        stripe_rect = (x0, y0, x1, y1)

    # Рисуем полоску
    stripe_color = random.randint(200, 230)
    draw.rectangle(stripe_rect, fill=stripe_color)

    # Рисуем элементы полоски (круги и зебру)
    draw_stripe_elements(draw, pos, orientation, stripe_width, stripe_length)

def draw_stripe_elements(draw, pos, orientation, stripe_width, stripe_length):
    """
    Рисует элементы полоски (черные круги и зебру)

    Args:
        draw: объект ImageDraw для рисования
        pos: позиция полоски (x, y)
        orientation: ориентация полоски ('horizontal' или 'vertical')
        stripe_width: ширина полоски
        stripe_length: длина полоски
    """
    if orientation == 'horizontal':
        # Черные круги слева
        num_circles = random.randint(1, 4)
        circle_radius = max(1, stripe_width // 4)  # Минимальный радиус 1
        circle_margin = max(1, stripe_width // 4)  # Минимальный отступ 1

        # Определяем количество рядов
        rows = 2 if num_circles > 2 else 1
        circles_per_row = (num_circles + 1) // 2 if rows == 2 else num_circles

        # Рисуем круги
        for row in range(rows):
            # Количество кругов в текущем ряду
            if row == 0:
                current_circles = circles_per_row
            else:
                current_circles = num_circles - circles_per_row

            # Координата Y для текущего ряда
            if rows == 1:
                circle_y = pos[1] + stripe_width // 2
            else:
                if row == 0:
                    circle_y = pos[1] + stripe_width // 3
                else:
                    circle_y = pos[1] + 2 * stripe_width // 3

            # Распределяем круги по горизонтали
            start_x = pos[0] + circle_radius + circle_margin

            for i in range(current_circles):
                circle_x = start_x + i * (2 * circle_radius + circle_margin)
                # Нормализуем координаты круга
                x0 = circle_x - circle_radius
                y0 = circle_y - circle_radius
                x1 = circle_x + circle_radius
                y1 = circle_y + circle_radius
                x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
                draw.ellipse([x0, y0, x1, y1], fill=0)

        # Рисуем зебру справа
        zebra_start_x = pos[0] + stripe_length - (stripe_length // 2)
        zebra_end_x = pos[0] + stripe_length - 5

        # Фиксированный шаг зебры
        stripe_step = 3
        current_x = zebra_start_x
        is_black = True

        while current_x < zebra_end_x:
            next_x = min(current_x + stripe_step, zebra_end_x)

            # Нормализуем координаты прямоугольника зебры
            x0 = current_x
            y0 = pos[1]
            x1 = next_x
            y1 = pos[1] + stripe_width
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)

            if x1 > x0 and y1 > y0:  # Проверяем, что прямоугольник имеет положительную площадь
                if is_black:
                    draw.rectangle([x0, y0, x1, y1], fill=0)
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=255)

            is_black = not is_black
            current_x = next_x

    else:  # Вертикальная ориентация
        # Черные круги сверху
        num_circles = random.randint(1, 4)
        circle_radius = max(1, stripe_width // 6)  # Минимальный радиус 1
        circle_margin = max(1, stripe_width // 6)  # Минимальный отступ 1

        # Определяем количество рядов
        rows = 2 if num_circles > 2 else 1
        circles_per_row = (num_circles + 1) // 2 if rows == 2 else num_circles

        # Рисуем круги
        for row in range(rows):
            # Количество кругов в текущем ряду
            if row == 0:
                current_circles = circles_per_row
            else:
                current_circles = num_circles - circles_per_row

            # Координата X для текущего ряда
            if rows == 1:
                circle_x = pos[0] + stripe_width // 2
            else:
                if row == 0:
                    circle_x = pos[0] + stripe_width // 3
                else:
                    circle_x = pos[0] + 2 * stripe_width // 3

            # Распределяем круги по вертикали
            start_y = pos[1] + circle_radius + circle_margin

            for i in range(current_circles):
                circle_y = start_y + i * (2 * circle_radius + circle_margin)
                # Нормализуем координаты круга
                x0 = circle_x - circle_radius
                y0 = circle_y - circle_radius
                x1 = circle_x + circle_radius
                y1 = circle_y + circle_radius
                x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
                draw.ellipse([x0, y0, x1, y1], fill=0)

        # Рисуем зебру снизу
        zebra_start_y = pos[1] + stripe_length - (stripe_length // 2)
        zebra_end_y = pos[1] + stripe_length - 5

        # Фиксированный шаг зебры
        stripe_step = 3
        current_y = zebra_start_y
        is_black = True

        while current_y < zebra_end_y:
            next_y = min(current_y + stripe_step, zebra_end_y)

            # Нормализуем координаты прямоугольника зебры
            x0 = pos[0]
            y0 = current_y
            x1 = pos[0] + stripe_width
            y1 = next_y
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)

            if x1 > x0 and y1 > y0:  # Проверяем, что прямоугольник имеет положительную площадь
                if is_black:
                    draw.rectangle([x0, y0, x1, y1], fill=0)
                else:
                    draw.rectangle([x0, y0, x1, y1], fill=255)

            is_black = not is_black
            current_y = next_y

def draw_a_seam(pattern_size=random.randint(15, 30), overlap_ratio=random.randint(18, 20) // 10,
                kernel_div=2, size=(512, 512), light_angle=None, light_intensity=None):
    """
    Рисует сварной шов из повторяющихся узоров в виде чешуек.
    Возвращает центры чешуек для последующего рисования непровара.

    Returns:
        image: основное изображение
        draw: объект для рисования поверх шва
        centers: список координат центров чешуек
        direction: направление шва
        start_point: начальная точка шва
        end_point: конечная точка шва
        pattern_size: размер узора
        seam_area: кортеж (x_min, y_min, x_max, y_max) - ограничивающий прямоугольник шва
    """

    # Генерируем случайные параметры освещения, если не заданы
    if light_angle is None:
        light_angle = random.uniform(0, 360)  # случайный угол от 0 до 360 градусов

    if light_intensity is None:
        light_intensity = random.uniform(0.1, 0.4)  # случайная интенсивность от 0.2 до 0.8

    gray = random.randint(10, 130)
    sharpness = random.randint(15, 90)
    start_point = 0
    end_point = 0

    # Добавляем параметр резкости чешуек
    seam_sharpness = random.uniform(0.3, 0.8)

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

    # Добавляем градиентные потемнения (пятна)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Создаем несколько случайных пятен потемнения
    num_spots = random.randint(2, 8)
    for _ in range(num_spots):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        spot_radius = random.randint(40, 200)
        max_darkness = random.randint(10, 20)

        for r in range(spot_radius, 0, -1):
            darkness = int(max_darkness * (1 - r / spot_radius) * (1 - r / spot_radius))
            alpha = int(255 * (1 - r / spot_radius))

            spot_layer = Image.new('L', size, 0)
            spot_draw = ImageDraw.Draw(spot_layer)
            # Нормализуем координаты пятна
            x0 = center_x - r
            y0 = center_y - r
            x1 = center_x + r
            y1 = center_y + r
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
            spot_draw.ellipse([x0, y0, x1, y1], fill=darkness)

            image = Image.composite(
                Image.new('L', size, max(0, gray - darkness)),
                image,
                spot_layer
            )

    # Создаем временный слой для чешуек
    seam_layer = Image.new('L', size, 0)
    seam_draw = ImageDraw.Draw(seam_layer)

    # Вычисляем шаг между чешуйками с учетом перекрытия
    step = int(pattern_size * (1 - overlap_ratio))
    if step < 1:
        step = 1

    choice = random.random()
    # Определяем направление шва
    if choice < 0.4:
        direction = 'circle'
    else:
        direction = random.choice(['horizontal', 'vertical','diagonal'])
    centers = []
    seam_fill = random.randint(140, 180)

    # Переменные для хранения параллельных линий (для кругового шва)
    parallel_line_1 = None
    parallel_line_2 = None
    n1 = 0  # Для кругового шва - количество центров первой дуги

    # Рисуем чешуйки в зависимости от направления
    if direction == 'horizontal':
        start_y = random.randint(pattern_size, height)
        start_point = (pattern_size, start_y)
        end_point = (width, start_y)

        for x in range(-pattern_size, width + pattern_size, step):
            y_offset = random.randint(-1, 1) * random.random()
            y = start_y + y_offset
            centers.append((x, y))
            radius = pattern_size / 2
            # Нормализуем координаты чешуйки
            x0 = x - radius
            y0 = y - radius
            x1 = x + radius
            y1 = y + radius
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
            seam_draw.ellipse([x0, y0, x1, y1],
                            fill=seam_fill, outline=seam_fill - 20)

    elif direction == 'vertical':
        start_x = random.randint(0, width)
        start_point = (start_x, 0)
        end_point = (start_x, height)

        for y in range(-pattern_size, height + pattern_size, step):
            x_offset = random.randint(-1, 1) * random.random()
            x = start_x + x_offset
            centers.append((x, y))
            radius = pattern_size / 2
            # Нормализуем координаты чешуйки
            x0 = x - radius
            y0 = y - radius
            x1 = x + radius
            y1 = y + radius
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
            seam_draw.ellipse([x0, y0, x1, y1],
                            fill=seam_fill, outline=seam_fill - 20)

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

        # Выбираем среднюю точку для создания параллельных линий
        middle_point = centers[len(centers) // 4]

        # Создаем две параллельные линии, которые будут образовывать контур трубы
        dx_dist = math.sqrt((central_pos1[0] - middle_point[0])**2 + (central_pos1[1] - middle_point[1])**2)

        parallel_line_1 = create_parallel_line(central_pos1[0], central_pos1[1], middle_point[0], middle_point[1], dx_dist/1.1, 2000)
        parallel_line_2 = create_parallel_line(central_pos1[0], central_pos1[1], middle_point[0], middle_point[1], -dx_dist/1.1, 2000)

        # Рисуем контур трубы
        width_line = random.randint(3, 9)
        seam_draw.line((parallel_line_1[0], parallel_line_1[1], parallel_line_1[2], parallel_line_1[3]), 150, width_line)
        seam_draw.line((parallel_line_2[0], parallel_line_2[1], parallel_line_2[2], parallel_line_2[3]), 150, width_line)

        # Теперь рисуем все чешуйки для обеих дуг
        radius_pattern = pattern_size / 2
        for center in centers:
            # Нормализуем координаты чешуйки
            x0 = center[0] - radius_pattern
            y0 = center[1] - radius_pattern
            x1 = center[0] + radius_pattern
            y1 = center[1] + radius_pattern
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
            seam_draw.ellipse([x0, y0, x1, y1],
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
            radius = pattern_size / 2
            # Нормализуем координаты чешуйки
            x0 = x - radius + offset_x
            y0 = y - radius + offset_y
            x1 = x + radius + offset_x
            y1 = y + radius + offset_y
            x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
            seam_draw.ellipse([x0, y0, x1, y1],
                         fill=seam_fill, outline=seam_fill - 20)

    # Рисуем белые центры (ядра) на слое чешуек
    white_centers = []
    if direction == 'circle':
        # Для кругового шва рисуем белые центры только для первой дуги
        for i in range(n1 - 1):
            white_centers.append(centers[i])
            white_centers.append(((centers[i][0] + centers[i + 1][0]) / 2,
                                  (centers[i][1] + centers[i + 1][1]) / 2))
    else:
        # Для других направлений как раньше
        for i in range(len(centers) - 1):
            white_centers.append(centers[i])
            white_centers.append(((centers[i][0] + centers[i + 1][0]) / 2,
                                  (centers[i][1] + centers[i + 1][1]) / 2))

    kernel_radius = pattern_size / (kernel_div * 2)
    for center_x, center_y in white_centers:
        # Нормализуем координаты ядра
        x0 = center_x - kernel_radius
        y0 = center_y - kernel_radius
        x1 = center_x + kernel_radius
        y1 = center_y + kernel_radius
        x0, y0, x1, y1 = normalize_rectangle_coords(x0, y0, x1, y1)
        seam_draw.ellipse([x0, y0, x1, y1],
                     fill=seam_fill + 30)

    # Применяем размытие к чешуйкам в зависимости от параметра резкости
    blur_radius = (1.0 - seam_sharpness) * 3.0
    if blur_radius > 0:
        seam_layer = seam_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Накладываем слой с чешуйками на основное изображение
    image = ImageChops.screen(image, seam_layer)

    # ВАЖНО: пересоздаем объект draw для основного изображения после наложения чешуек
    draw = ImageDraw.Draw(image)

    # Вычисляем область шва (bounding box)
    parallel_lines = (parallel_line_1, parallel_line_2) if direction == 'circle' else None
    seam_area = calculate_seam_area(centers, direction, pattern_size, start_point, end_point, parallel_lines)

    # Всегда добавляем случайный текст (гарантируем минимум 3 текста)
    draw_random_text(draw, seam_area, pattern_size, gray, size)

    # Всегда рисуем светлую полоску (эталон чувствительности)
    draw_sensitivity_stripe(draw, seam_area, size)

    # Применяем градиент освещения
    image = apply_lighting_gradient(image, light_angle, light_intensity)

    # После применения освещения нужно пересоздать draw
    draw = ImageDraw.Draw(image)

    return image, draw, centers, direction, start_point, end_point, pattern_size, seam_area
