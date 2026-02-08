from PIL import Image, ImageDraw, ImageFilter, ImageChops
import random
import math
import datetime
from src.generators.generate_seam import draw_a_seam
import statistics
import json
from tqdm import tqdm
import time


def _draw_defect_centered(image, centers, defect_width, accuracy=0.9, defect_sharpness=1.0,
                         start_position=None, left_length=None, right_length=None):
    """
    Рисует непровар вдоль шва с эффектом истёртости и плавными переходами

    Args:
        image: основное изображение
        centers: список координат центров чешуек шва
        defect_width: базовая ширина непровара
        accuracy: точность выполнения (0.0-1.0), где 1.0 - идеальная линия
        defect_sharpness: резкость непровара (0.6-1.0), где 1.0 - максимальная четкость
        start_position: позиция начала непровара вдоль шва (0.0-1.0), None - случайная
        left_length: длина левой ветви в долях от длины шва (0.0-1.0), None - случайная
        right_length: длина правой ветви в долях от длины шва (0.0-1.0), None - случайная
    """
    if not centers or len(centers) < 2:
        return image

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
        return image

    # Определяем позицию начала непровара
    if start_position is None:
        start_position = random.random()  # Случайная позиция вдоль шва
    start_distance = start_position * total_length

    # Определяем длины ветвей
    if left_length is None:
        left_length = random.uniform(0.1, 1.0)  # Случайная длина левой ветви
    if right_length is None:
        right_length = random.uniform(0.1, 1.0)  # Случайная длина правой ветви

    # Находим начальную точку на шве
    current_distance = 0
    start_point = None
    for i in range(len(centers) - 1):
        segment_length = segment_lengths[i]
        if current_distance <= start_distance <= current_distance + segment_length:
            # Находим точку внутри сегмента
            t = (start_distance - current_distance) / segment_length
            start_x = centers[i][0] + t * (centers[i + 1][0] - centers[i][0])
            start_y = centers[i][1] + t * (centers[i + 1][1] - centers[i][1])
            start_point = (start_x, start_y)
            start_segment_index = i
            start_t = t
            break
        current_distance += segment_length

    if start_point is None:
        # Если не нашли начальную точку, используем последнюю точку
        start_point = centers[-1]
        start_segment_index = len(centers) - 2
        start_t = 1.0

    # Создаём контрольные точки для плавного шума
    num_control_points = max(3, int(total_length / 5))
    control_points = [random.random() for _ in range(num_control_points)]

    # Создаем контрольные точки для толщины
    thickness_points = [random.uniform(0.7, 1.3) for _ in range(num_control_points)]

    # Функция для получения значения шума в позиции t (0-1)
    def get_noise(t):
        idx = t * (num_control_points - 1)
        idx0 = int(idx)
        idx1 = min(num_control_points - 1, idx0 + 1)
        frac = idx - idx0
        return control_points[idx0] * (1 - frac) + control_points[idx1] * frac

    # Функция для получения значения толщины в позиции t (0-1)
    def get_thickness(t):
        idx = t * (num_control_points - 1)
        idx0 = int(idx)
        idx1 = min(num_control_points - 1, idx0 + 1)
        frac = idx - idx0
        return thickness_points[idx0] * (1 - frac) + thickness_points[idx1] * frac

    # Создаем временный слой для непровара
    image_size = image.size
    defect_layer = Image.new('L', image_size, 0)
    defect_draw = ImageDraw.Draw(defect_layer)

    # Список для отслеживания всех точек, где был нарисован дефект
    defect_points = []

    # Функция для рисования ветви непровара
    def draw_branch(start_segment, start_t_param, direction, max_length):
        nonlocal prev_draw, prev_x, prev_y, prev_width

        current_segment = start_segment
        current_t = start_t_param
        current_distance_local = 0
        segment_length_remaining = segment_lengths[current_segment] * (1 - current_t) if direction == 1 else segment_lengths[current_segment] * current_t

        while current_distance_local < max_length and 0 <= current_segment < len(segment_lengths):
            if direction == 1:  # Вправо (к концу шва)
                start_x, start_y = centers[current_segment]
                end_x, end_y = centers[current_segment + 1]
                segment_length_full = segment_lengths[current_segment]

                # Определяем диапазон t для текущего сегмента
                t_start = current_t
                t_end = 1.0

                # Ограничиваем конечную точку, если превышаем максимальную длину
                if current_distance_local + segment_length_remaining > max_length:
                    t_end = t_start + (max_length - current_distance_local) / segment_length_full
                    segment_length_remaining = 0
                else:
                    segment_length_remaining = segment_lengths[current_segment + 1] if current_segment + 1 < len(segment_lengths) else 0

            else:  # Влево (к началу шва)
                start_x, start_y = centers[current_segment + 1] if current_segment + 1 < len(centers) else centers[current_segment]
                end_x, end_y = centers[current_segment]
                segment_length_full = segment_lengths[current_segment]

                # Определяем диапазон t для текущего сегмента
                t_start = 1 - current_t
                t_end = 0.0

                # Ограничиваем конечную точку, если превышаем максимальную длину
                if current_distance_local + segment_length_remaining > max_length:
                    t_end = t_start - (max_length - current_distance_local) / segment_length_full
                    segment_length_remaining = 0
                else:
                    segment_length_remaining = segment_lengths[current_segment - 1] if current_segment - 1 >= 0 else 0

            # Проходим по текущему сегменту
            num_substeps = max(5, int(segment_length_full * abs(t_end - t_start) / 2))
            for j in range(num_substeps + 1):
                if direction == 1:
                    t_segment = t_start + (t_end - t_start) * (j / num_substeps)
                else:
                    t_segment = t_start + (t_end - t_start) * (j / num_substeps)

                # Вычисляем координаты точки
                if direction == 1:
                    x = start_x + t_segment * (end_x - start_x)
                    y = start_y + t_segment * (end_y - start_y)
                else:
                    x = start_x + t_segment * (end_x - start_x)
                    y = start_y + t_segment * (end_y - start_y)

                # Общая позиция вдоль всей ветви (0-1)
                t_branch = current_distance_local / max_length

                # Получаем значение шума для этой позиции
                noise = get_noise(t_branch) - random.random() / 10

                # Получаем множитель толщины для этой позиции
                thickness_factor = get_thickness(t_branch)

                # Определяем, нужно ли рисовать в этой позиции
                should_draw = noise < accuracy

                # Определяем толщину в этой позиции с плавным изменением
                if should_draw:
                    # Базовый расчет толщины с плавными изменениями
                    base_width = defect_width * thickness_factor

                    # Плавное изменение толщины между точками
                    if prev_draw:
                        # Интерполяция между предыдущей и текущей толщиной
                        t_width = j / num_substeps
                        current_width = prev_width * (1 - t_width) + base_width * t_width
                    else:
                        current_width = base_width

                    # Добавляем небольшие неровности к толщине
                    current_width *= random.uniform(0.9, 1.1)
                    current_width = max(1, current_width)

                    # Рисуем сегмент на временном слое
                    if prev_draw:
                        # Плавный переход от предыдущего сегмента
                        defect_draw.line([(prev_x, prev_y), (x, y)], fill=255,
                                  width=int(current_width), joint="curve")
                    else:
                        # Начинаем новый сегмент
                        radius = current_width / 2
                        defect_draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                                     fill=255)

                    # Добавляем точку в список дефектных точек
                    defect_points.append((x, y, current_width))

                    # Сохраняем текущую толщину для следующей итерации
                    prev_width = current_width
                elif prev_draw:
                    # Завершаем предыдущий сегмент
                    radius = prev_width / 2
                    defect_draw.ellipse([prev_x - radius, prev_y - radius, prev_x + radius, prev_y + radius],
                                 fill=255)
                    # Добавляем точку завершения
                    defect_points.append((prev_x, prev_y, prev_width))

                prev_draw = should_draw
                prev_x, prev_y = x, y

            # Обновляем расстояние и переходим к следующему сегменту
            if direction == 1:
                current_distance_local += segment_length_full * abs(t_end - t_start)
                current_segment += 1
                current_t = 0.0
            else:
                current_distance_local += segment_length_full * abs(t_end - t_start)
                current_segment -= 1
                current_t = 1.0

    # Инициализируем переменные для рисования
    prev_draw = False
    prev_x, prev_y = start_point
    prev_width = defect_width * get_thickness(0)

    # Рисуем левую ветвь (к началу шва)
    left_max_length = left_length * total_length
    if left_max_length > 0:
        draw_branch(start_segment_index, start_t, -1, left_max_length)

    # Сбрасываем состояние для правой ветви
    prev_draw = False
    prev_x, prev_y = start_point
    prev_width = defect_width * get_thickness(0)

    # Рисуем правую ветвь (к концу шва)
    right_max_length = right_length * total_length
    if right_max_length > 0:
        draw_branch(start_segment_index, start_t, 1, right_max_length)

    # Завершаем последний сегмент, если нужно
    if prev_draw:
        radius = prev_width / 2
        defect_draw.ellipse([prev_x - radius, prev_y - radius, prev_x + radius, prev_y + radius],
                     fill=255)
        # Добавляем точку завершения
        defect_points.append((prev_x, prev_y, prev_width))

    # Применяем размытие к непровару в зависимости от параметра резкости
    blur_radius = (1.0 - defect_sharpness) * 3.0
    if blur_radius > 0:
        defect_layer = defect_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    # Накладываем слой с непроваром на основное изображение
    # Инвертируем слой непровара для затемнения
    defect_layer = ImageChops.invert(defect_layer)
    image = ImageChops.darker(image, defect_layer)

    # ВЫЧИСЛЯЕМ ТОЧНУЮ ОБЛАСТЬ ДЕФЕКТА
    if defect_points:
        # Находим минимальные и максимальные координаты всех точек дефекта
        # Учитываем ширину дефекта в каждой точке
        min_x = min([x - w/2 for x, y, w in defect_points])
        max_x = max([x + w/2 for x, y, w in defect_points])
        min_y = min([y - w/2 for x, y, w in defect_points])
        max_y = max([y + w/2 for x, y, w in defect_points])

        # Добавляем отступ для учета размытия
        padding = blur_radius * 2
        target_area = [
            max(0, min_x - padding),
            max(0, min_y - padding),
            min(image_size[0], max_x + padding),
            min(image_size[1], max_y + padding)
        ]
    else:
        # Если точек дефекта нет, используем область вокруг начальной точки
        padding = defect_width * 5
        target_area = [
            max(0, start_point[0] - padding),
            max(0, start_point[1] - padding),
            min(image_size[0], start_point[0] + padding),
            min(image_size[1], start_point[1] + padding)
        ]

    return image, {f"D_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}":target_area}


def make_incomplete_fusion():
    # Рисуем шов и получаем центры чешуек и граничные точки
    result, draw, centers, direction, start_point, end_point, pattern_size, seam_area = draw_a_seam()

    # Параметры непровара
    defect_width = random.randint(2, 3)
    defect_sharpness = random.uniform(0.2, 0.7)  # Случайная резкость от 0.6 до 1.0

    # Случайные параметры для позиции и длины непровара
    start_position = random.random()  # Случайная позиция вдоль шва
    left_length = random.uniform(0.1, 1.0)  # Случайная длина левой ветви
    right_length = random.uniform(0.1, 1.0)  # Случайная длина правой ветви

    # Рисуем непровар вдоль шва
    result, data = _draw_defect_centered(
        result, centers, defect_width,
        random.randint(6,6)/10, defect_sharpness,
        start_position, left_length, right_length
    )

    return result, data


def test():
    # Для тестирования можно задать конкретные значения
    result, data = make_incomplete_fusion()

    # Или вручную задать параметры:
    # result, draw, centers, direction, start_point, end_point, pattern_size = draw_a_seam()
    # result, data = _draw_defect_centered(
    #     result, centers, 2, 0.6, 0.5,
    #     start_position=0.5, left_length=0.2, right_length=0.2
    # )

    draw = ImageDraw.Draw(result)
    data_item = list(data.items())[0]
    # Рисуем ограничивающий прямоугольник для проверки
    bbox = data_item[1]
    draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=2)
    result.show()


def generate_dataset(directory, num_images=100):
    summary = []
    progress_bar = tqdm(total=num_images)
    path = "C:/Users/Алексей/Documents/XVL_2025_set"
    for i in range(num_images):
        fusion_image, data = make_incomplete_fusion()

        # Сохраняем изображение

        fusion_image.save(f"{path}/{directory}/N{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

        # Сохраняем метаданные
        data_item = list(data.items())[0]
        metadata = {
            "image_id": f"N{i}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
            "defect_id": data_item[0],
            "target_area": data_item[1]
        }
        summary.append(metadata)

        progress_bar.set_description(f"Генерация непровара № {i+1}")
        progress_bar.update(1)

    with open(f"{directory}/summary.json","w") as file:
        json.dump(summary, file, indent=2)
