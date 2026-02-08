import random
import datetime
import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import sys
import numpy as np
import copy
import time, psutil
import cv2
import math

# Добавляем путь к модулям с функциями
sys.path.append('.')

# Импортируем ваши функции
try:
    from crack import make_a_crack
    from incomplete_fusion import make_incomplete_fusion
    from pore import make_pore
    from empty import make_empty_seam
    print("Все модули успешно импортированы")
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Убедитесь, что все файлы находятся в той же директории:")
    print("- crack.py")
    print("- incomplete_fusion.py")
    print("- pore.py")
    print("- empty_seam.py")
    exit()

def get_bbox_rectangle(bbox_data):
    """
    Преобразует различные форматы bbox в прямоугольник [x1, y1, x2, y2]
    """
    if isinstance(bbox_data[0], (list, tuple)):
        # Многоугольник (для непровара) - берем ограничивающий прямоугольник
        x_coords = [p[0] for p in bbox_data]
        y_coords = [p[1] for p in bbox_data]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
    elif len(bbox_data) == 2:
        # Точка (для пор) - создаем маленький прямоугольник вокруг точки
        x, y = bbox_data
        size = 5  # размер для пор
        return [x-size, y-size, x+size, y+size]
    elif len(bbox_data) == 4:
        # Уже прямоугольник (для трещин)
        return bbox_data
    else:
        raise ValueError(f"Неизвестный формат bbox: {bbox_data}")

def normalize_bbox(bbox):
    """
    Нормализует bbox, чтобы x1 <= x2 и y1 <= y2
    """
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        return [x1, y1, x2, y2]
    return bbox

def load_real_samples_without_padding(real_samples_dir="C:/PycharmProjects/XVL/src/generators/real_samples"):
    """
    Загружает реальные изображения и их аннотации БЕЗ добавления серых полей
    Просто обрезает изображение до фактического контента
    """
    annotations_path = real_samples_dir + "/annotations.json"

    if not os.path.exists(annotations_path):
        print(f"Файл аннотаций не найден: {annotations_path}")
        return [], {}

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    real_images = []
    real_bboxes = {}

    for img_name in annotations.keys():
        img_path = real_samples_dir + f"/{img_name}.jpg"

        try:
            # Открываем изображение
            img = Image.open(img_path).convert('L')  # Конвертируем в одноканальное

            print(f"Загружаем изображение: {img_name}")
            print(f"  Оригинальный размер: {img.size}")

            # Находим фактические границы изображения (не серые)
            img_array = np.array(img)

            # Определяем порог для серого фона (типичное значение для серого фона)
            bg_threshold = 110

            # Создаем маску для ненулевых пикселей
            mask = img_array < bg_threshold

            if np.any(mask):
                # Находим границы маски
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)

                if np.any(rows) and np.any(cols):
                    y_min, y_max = np.where(rows)[0][[0, -1]]
                    x_min, x_max = np.where(cols)[0][[0, -1]]

                    # Обрезаем изображение
                    cropped_img = img.crop((x_min, y_min, x_max + 1, y_max + 1))

                    # Корректируем bbox с учетом обрезки
                    original_bboxes = annotations[img_name]
                    corrected_bboxes = {}

                    for defect_name, bbox_data in original_bboxes.items():
                        corrected_bbox = correct_bbox_for_crop_simple(bbox_data, x_min, y_min)
                        if corrected_bbox is not None:
                            corrected_bboxes[defect_name] = corrected_bbox

                    # Сохраняем обрезанное изображение и скорректированные bbox
                    real_images.append((img_name, cropped_img))
                    real_bboxes[img_name] = corrected_bboxes

                    print(f"  Обрезанный размер: {cropped_img.size}")
                    print(f"  Смещение обрезки: x={x_min}, y={y_min}")
                    print(f"  Осталось дефектов: {len(corrected_bboxes)}")
                else:
                    # Если маска пустая, используем оригинальное изображение
                    real_images.append((img_name, img))
                    real_bboxes[img_name] = annotations[img_name]
                    print(f"  Обрезка не требуется, оставляем оригинальный размер")
            else:
                # Если нет дефектов, используем оригинальное изображение
                real_images.append((img_name, img))
                real_bboxes[img_name] = annotations[img_name]
                print(f"  Нет дефектов для обрезки, оставляем оригинальный размер")

        except Exception as e:
            print(f"Ошибка при загрузке изображения {img_name}: {e}")
            continue

    print(f"\nВсего загружено реальных изображений: {len(real_images)}")
    return real_images, real_bboxes

def correct_bbox_for_crop_simple(bbox_data, x_offset, y_offset):
    """
    Простая коррекция координат bbox после обрезки изображения
    """
    if isinstance(bbox_data[0], (list, tuple)):
        # Многоугольник
        corrected_polygon = []
        for point in bbox_data:
            if len(point) == 2:
                x, y = point
                new_x = x - x_offset
                new_y = y - y_offset
                corrected_polygon.append([int(new_x), int(new_y)])
        return corrected_polygon if len(corrected_polygon) >= 3 else None

    elif len(bbox_data) == 2:
        # Точка
        x, y = bbox_data
        new_x = x - x_offset
        new_y = y - y_offset
        return [int(new_x), int(new_y)]

    elif len(bbox_data) == 4:
        # Прямоугольник
        x1, y1, x2, y2 = bbox_data
        new_x1 = x1 - x_offset
        new_y1 = y1 - y_offset
        new_x2 = x2 - x_offset
        new_y2 = y2 - y_offset
        return [int(new_x1), int(new_y1), int(new_x2), int(new_y2)]

    return bbox_data

def augment_real_image_simple(image, bboxes_dict, brightness_range=(80, 120), noise_level_range=(0.01, 0.03)):
    """
    Простая аугментация реального изображения (без вращения)
    Не меняет размер изображения
    """
    augmented_image = image.copy()
    augmented_bboxes = copy.deepcopy(bboxes_dict)

    # 1. Изменение яркости
    brightness_factor = random.uniform(brightness_range[0]/100.0, brightness_range[1]/100.0)
    enhancer = ImageEnhance.Brightness(augmented_image)
    augmented_image = enhancer.enhance(brightness_factor)

    # 3. Добавление зернистости (шума)
    noise_level = random.uniform(noise_level_range[0], noise_level_range[1])
    img_array = np.array(augmented_image, dtype=np.float32)
    noise = np.random.normal(0, noise_level * 255, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
    augmented_image = Image.fromarray(img_array)

    return augmented_image, augmented_bboxes

def apply_random_blur(image, min_blur=0, max_blur=2):
    """
    Применяет случайное размытие (мыльный эффект) к изображению
    0 - абсолютно чёткое, больше - сильнее размытие
    """
    # Выбираем случайный коэффициент размытия
    blur_radius = random.uniform(min_blur, max_blur)

    # Если размытие 0 - возвращаем исходное изображение
    if blur_radius <= 0.1:  # Почти 0
        return image

    # Применяем размытие Гаусса
    # Преобразуем радиус в целое число для GaussianBlur
    blur_radius_int = max(1, int(blur_radius))

    # Применяем размытие
    blurred_image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius_int))

    return blurred_image

def draw_safe_rectangle(draw, bbox, outline="red", width=2):
    """
    Безопасное рисование прямоугольника с проверкой координат
    """
    if len(bbox) == 4:
        x1, y1, x2, y2 = bbox

        # Нормализуем координаты
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        # Рисуем только если координаты валидны
        if x2 > x1 and y2 > y1:
            draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
            return True
    return False

def save_dataset(directory="./data/training/train", num_images=100, original_step=5,
                 min_blur=0, max_blur=2):
    """
    Сохраняет датасет изображений с дефектами и создает annotations.json
    original_step - частота реальных изображений (каждое original_step-ое изображение - реальное)
    min_blur, max_blur - диапазон размытия (0 - абсолютно чёткое)
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    # Загружаем реальные изображения БЕЗ добавления полей
    real_images, real_bboxes = load_real_samples_without_padding()

    # Словарь для аннотаций
    annotations = {}

    # Счетчики
    counter = 1
    progress_bar = tqdm(total=num_images, desc="Сохранение датасета")

    for i in range(num_images):
        # Определяем, будет ли это реальное изображение
        is_real = (original_step > 0 and real_images and (i % original_step == 0))

        if is_real and real_images:
            # Используем реальное изображение с аугментацией
            img_idx = i // original_step % len(real_images)
            img_name, image = real_images[img_idx]
            bboxes = real_bboxes.get(img_name, {})

            # Аугментируем реальное изображение (без изменения размера)
            image, bbox_dict = augment_real_image_simple(image, bboxes)

            # Применяем случайное размытие
            image = apply_random_blur(image, min_blur, max_blur)

            # Формируем аннотации
            image_annotations = {}
            defect_counters = {}

            for defect_name, bbox in bbox_dict.items():
                # Определяем тип дефекта
                if "Непровар" in defect_name:
                    defect_type = "Непровар"
                elif "Трещина" in defect_name:
                    defect_type = "Трещина"
                elif "Скопление_пор" in defect_name:
                    defect_type = "Скопление_пор"
                elif "Одиночное_включение" in defect_name:
                    defect_type = "Одиночное_включение"
                else:
                    defect_type = defect_name.split("_")[0] if "_" in defect_name else defect_name

                # Обновляем счетчик
                if defect_type not in defect_counters:
                    defect_counters[defect_type] = 1

                # Формируем имя дефекта
                new_defect_name = f"{defect_type}_№{defect_counters[defect_type]}"
                defect_counters[defect_type] += 1

                # Нормализуем bbox
                bbox = normalize_bbox(bbox)
                image_annotations[new_defect_name] = [int(coord) for coord in bbox]

            # Добавляем в общие аннотации
            image_id = str(counter)
            annotations[image_id] = image_annotations

            # Сохраняем изображение
            image_filename = f"{image_id}.png"
            image_path = os.path.join(directory, image_filename)
            image.save(image_path)

        else:
            # Генерируем синтетическое изображение
            weights = [0.2, 0.2, 0.15, 0.15, 0.3]  # 30% пустых изображений
            defect_type = random.choices(
                ["crack", "incomplete_fusion", "single_pore", "cluster_pores", "empty"],
                weights=weights
            )[0]

            # Генерируем изображение
            if defect_type == "crack":
                crack_type = random.choice(["single", "tree", "shattered", "transverse"])
                image, bbox_dict = make_a_crack(crack_type)
                defect_label = "Трещина"
            elif defect_type == "incomplete_fusion":
                image, bbox_dict = make_incomplete_fusion()
                defect_label = "Непровар"
            elif defect_type == "single_pore":
                image, bbox_dict = make_pore(num_pores=1)
                defect_label = "Одиночное_включение"
            elif defect_type == "cluster_pores":
                num_pores = random.randint(5, 32)
                image, bbox_dict = make_pore(num_pores=num_pores)
                defect_label = "Скопление_пор"
            else:  # empty
                image = make_empty_seam()
                bbox_dict = {}
                defect_label = "empty"

            # Применяем случайное размытие
            image = apply_random_blur(image, min_blur, max_blur)

            # Формируем аннотации
            image_annotations = {}

            if bbox_dict:
                defect_counters = {
                    "Непровар": 1,
                    "Одиночное_включение": 1,
                    "Скопление_пор": 1,
                    "Трещина": 1
                }

                pore_type = defect_label if defect_label in ["Одиночное_включение", "Скопление_пор"] else None

                for defect_id, bbox_data in bbox_dict.items():
                    # Получаем прямоугольник bbox
                    bbox_rect = get_bbox_rectangle(bbox_data)
                    bbox_rect = [int(round(coord)) for coord in bbox_rect]
                    bbox_rect = normalize_bbox(bbox_rect)

                    # Определяем имя дефекта
                    if defect_label == "Трещина":
                        defect_name = f"{defect_label}_№{defect_counters['Трещина']}"
                        defect_counters['Трещина'] += 1
                    elif defect_label == "Непровар":
                        defect_name = f"{defect_label}_№{defect_counters['Непровар']}"
                        defect_counters['Непровар'] += 1
                    elif pore_type == "Одиночное_включение":
                        defect_name = f"{pore_type}_№{defect_counters['Одиночное_включение']}"
                        defect_counters['Одиночное_включение'] += 1
                    elif pore_type == "Скопление_пор":
                        defect_name = f"{pore_type}_№{defect_counters['Скопление_пор']}"
                        defect_counters['Скопление_пор'] += 1
                    else:
                        defect_name = f"Дефект_№{defect_counters.get(defect_label, 1)}"

                    # Добавляем в аннотации
                    image_annotations[defect_name] = bbox_rect

            # Добавляем в общие аннотации
            image_id = str(counter)
            annotations[image_id] = image_annotations

            # Сохраняем изображение
            image_filename = f"{image_id}.png"
            image_path = os.path.join(directory, image_filename)
            image.save(image_path)

        # Увеличиваем счетчик
        counter += 1
        progress_bar.update(1)

    progress_bar.close()

    # Сохраняем аннотации в JSON файл
    annotations_file = os.path.join(directory, "annotations.json")
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"\nДатасет успешно сохранен в {directory}")
    print(f"Всего изображений: {num_images}")
    print(f"Диапазон размытия: {min_blur}-{max_blur}")
    print(f"Размер реальных изображений сохранен без изменений")

def demo_dataset(num_images=20, original_step=5, min_blur=0, max_blur=2):
    """
    Демонстрирует генерацию в реальном времени с отображением bbox
    min_blur, max_blur - диапазон размытия (0 - абсолютно чёткое)
    """
    print("\n" + "="*60)
    print("ДЕМОНСТРАЦИЯ ГЕНЕРАЦИИ В РЕАЛЬНОМ ВРЕМЕНИ")
    print("="*60)

    # Загружаем реальные изображения БЕЗ добавления полей
    real_images, real_bboxes = load_real_samples_without_padding()

    print(f"\nДемонстрация {num_images} изображений")
    print(f"Шаг реальных изображений: {original_step}")
    print(f"Диапазон размытия: {min_blur}-{max_blur}")
    print("\nИнструкция:")
    print("- Каждое изображение будет показываться 3 секунды")
    print("- Нажмите любую клавишу, чтобы перейти к следующему")
    print("- Нажмите 'q', чтобы выйти из демонстрации")
    print("\nНачинаем через 2 секунды...")
    time.sleep(2)

    counter = 1
    for i in range(num_images):
        # Определяем, будет ли это реальное изображение
        is_real = (original_step > 0 and real_images and (i % original_step == 0))

        if is_real and real_images:
            # Используем реальное изображение с аугментацией
            img_idx = i // original_step % len(real_images)
            img_name, image = real_images[img_idx]
            bboxes = real_bboxes.get(img_name, {})

            # Аугментируем реальное изображение (без изменения размера)
            image, bbox_dict = augment_real_image_simple(image, bboxes)

            # Применяем случайное размытие
            image = apply_random_blur(image, min_blur, max_blur)

            print(f"\nИзображение {counter}: РЕАЛЬНОЕ (аугментированное)")
            print(f"  Оригинальное имя: {img_name}")
            print(f"  Размер: {image.size}")
            print(f"  Дефектов: {len(bbox_dict)}")

            # Рисуем bbox на изображении
            image_with_bbox = image.copy()
            draw = ImageDraw.Draw(image_with_bbox)
            for defect_name, bbox in bbox_dict.items():
                if len(bbox) == 4:
                    draw_safe_rectangle(draw, bbox, outline="red", width=2)
                elif len(bbox) == 2:
                    x, y = bbox
                    draw.ellipse([x-3, y-3, x+3, y+3], outline="green", width=2)
                elif isinstance(bbox[0], (list, tuple)):
                    draw.polygon(bbox, outline="blue", width=2)

            image = image_with_bbox

        else:
            # Генерируем синтетическое изображение
            weights = [0.2, 0.2, 0.15, 0.15, 0.3]
            defect_type = random.choices(
                ["crack", "incomplete_fusion", "single_pore", "cluster_pores", "empty"],
                weights=weights
            )[0]

            # Генерируем изображение
            if defect_type == "crack":
                crack_type = random.choice(["single", "tree", "shattered", "transverse"])
                image, bbox_dict = make_a_crack(crack_type)
                defect_label = "Трещина"
            elif defect_type == "incomplete_fusion":
                image, bbox_dict = make_incomplete_fusion()
                defect_label = "Непровар"
            elif defect_type == "single_pore":
                image, bbox_dict = make_pore(num_pores=1)
                defect_label = "Одиночное_включение"
            elif defect_type == "cluster_pores":
                num_pores = random.randint(5, 32)
                image, bbox_dict = make_pore(num_pores=num_pores)
                defect_label = "Скопление_пор"
            else:  # empty
                image = make_empty_seam()
                bbox_dict = {}
                defect_label = "empty"

            # Применяем случайное размытие
            image = apply_random_blur(image, min_blur, max_blur)

            print(f"\nИзображение {counter}: СИНТЕТИЧЕСКОЕ")
            print(f"  Тип дефекта: {defect_label}")
            print(f"  Размер: {image.size}")
            print(f"  Дефектов: {len(bbox_dict)}")

            # Рисуем bbox на изображении
            if bbox_dict:
                image_with_bbox = image.copy()
                draw = ImageDraw.Draw(image_with_bbox)
                for defect_id, bbox_data in bbox_dict.items():
                    bbox_rect = get_bbox_rectangle(bbox_data)
                    bbox_rect = [int(round(coord)) for coord in bbox_rect]
                    bbox_rect = normalize_bbox(bbox_rect)
                    draw_safe_rectangle(draw, bbox_rect, outline="red", width=2)
                image = image_with_bbox

        # Показываем изображение
        try:
            # Конвертируем PIL Image в numpy array для OpenCV
            if image.mode == 'L':
                image_rgb = image.convert('RGB')
                cv_image = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
            else:
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            window_title = f"Демонстрация - Изображение {counter}/{num_images}"
            cv2.imshow(window_title, cv_image)

            # Ждем 3 секунды или нажатие клавиши
            key = cv2.waitKey(3000) & 0xFF

            cv2.destroyAllWindows()

            if key == ord('q'):
                print("\nДемонстрация прервана пользователем")
                break

        except Exception as e:
            print(f"Ошибка при отображении изображения: {e}")

        counter += 1

    cv2.destroyAllWindows()
    print(f"\nДемонстрация завершена!")
    print(f"Всего показано изображений: {counter-1}")

def main():
    """
    Главное меню для выбора функции
    """
    print("Генератор синтетического датасета дефектов сварных швов")
    print("=" * 50)
    print("1. Сохранить датасет")
    print("2. Демонстрация в реальном времени")
    print("3. Выход")

    choice = input("\nВыберите действие (1-3): ")

    if choice == "1":
        try:
            num_images = int(input("Введите количество изображений (по умолчанию 100): ") or "100")
            original_step = int(input("Введите шаг для реальных изображений (0 - только синтетические, по умолчанию 5): ") or "5")
            output_dir = input("Введите директорию для сохранения (по умолчанию: ./dataset): ") or "./dataset"

            # Запрашиваем параметры размытия
            min_blur_input = input("Введите минимальное размытие (0 - чёткое, по умолчанию 0): ") or "0"
            max_blur_input = input("Введите максимальное размытие (по умолчанию 2): ") or "2"

            min_blur = float(min_blur_input)
            max_blur = float(max_blur_input)

            save_dataset(directory=output_dir, num_images=num_images,
                        original_step=original_step, min_blur=min_blur, max_blur=max_blur)

        except ValueError as e:
            print(f"Ошибка ввода: {e}")
            print("Используются значения по умолчанию.")
            save_dataset()

    elif choice == "2":
        try:
            num_images = int(input("Введите количество изображений (по умолчанию 20): ") or "20")
            original_step = int(input("Введите шаг для реальных изображений (0 - только синтетические, по умолчанию 5): ") or "5")

            # Запрашиваем параметры размытия
            min_blur_input = input("Введите минимальное размытие (0 - чёткое, по умолчанию 0): ") or "0"
            max_blur_input = input("Введите максимальное размытие (по умолчанию 2): ") or "2"

            min_blur = float(min_blur_input)
            max_blur = float(max_blur_input)

            demo_dataset(num_images=num_images, original_step=original_step,
                        min_blur=min_blur, max_blur=max_blur)

        except ValueError as e:
            print(f"Ошибка ввода: {e}")
            print("Используются значения по умолчанию.")
            demo_dataset()

    elif choice == "3":
        print("Выход")

    else:
        print("Неверный выбор")

if __name__ == "__main__":
    main()