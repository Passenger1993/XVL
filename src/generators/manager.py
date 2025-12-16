import random
import datetime
import json
import os
from tqdm import tqdm
from PIL import Image, ImageDraw
import sys

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
        size = 10  # размер для пор
        return [x-size, y-size, x+size, y+size]
    elif len(bbox_data) == 4:
        # Уже прямоугольник (для трещин)
        return bbox_data
    else:
        raise ValueError(f"Неизвестный формат bbox: {bbox_data}")


def generate_dataset(directory="synthetic_dataset", num_images=100):
    """
    Генерирует датасет изображений с дефектами и создает annotations.json
    """
    # Создаем директории
    images_dir = os.path.join(directory, "train")

    if not os.path.exists(directory):
        os.makedirs(images_dir, exist_ok=True)

    # Словарь для аннотаций
    annotations = {}

    # Цвета для разных типов дефектов (для визуализации)
    defect_colors = {
        "crack": "red",
        "incomplete_fusion": "blue",
        "pore": "green",
        "empty": "gray"
    }

    progress_bar = tqdm(total=num_images, desc="Генерация датасета")

    for i in range(num_images):
        # Выбираем случайный тип дефекта
        weights = [0.2, 0.2, 0.15, 0.15, 0.3]  # 30% пустых изображений
        defect_type = random.choices(
            ["crack", "incomplete_fusion", "single_pore", "cluster_pores", "empty"],
            weights=weights
        )[0]

        # Генерируем изображение
        if defect_type == "crack":
            # Случайный тип трещины
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

        # Сохраняем изображение
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_id = f"Н{i+1}"
        image_filename = f"{image_id}.png"
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path)

        # Если есть дефекты, добавляем их в аннотации
        if bbox_dict:
            image_annotations = {}

            # Счетчики для каждого типа дефекта в этом изображении
            defect_counters = {
                "Непровар": 1,
                "Одиночное_включение": 1,
                "Скопление_пор": 1,
                "Трещина": 1
            }

            # Для пор нужно различать одиночные и скопления
            if defect_label in ["Одиночное_включение", "Скопление_пор"]:
                pore_type = defect_label
            else:
                pore_type = None

            for defect_id, bbox_data in bbox_dict.items():
                # Получаем прямоугольник bbox
                bbox_rect = get_bbox_rectangle(bbox_data)

                # Округляем координаты до целых чисел
                bbox_rect = [int(round(coord)) for coord in bbox_rect]

                # Убедимся, что координаты в пределах изображения
                bbox_rect[0] = max(0, bbox_rect[0])
                bbox_rect[1] = max(0, bbox_rect[1])
                bbox_rect[2] = min(image.width, bbox_rect[2])
                bbox_rect[3] = min(image.height, bbox_rect[3])

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
                    # На всякий случай
                    defect_name = f"Дефект_№{defect_counters.get(defect_label, 1)}"

                # Добавляем в аннотации изображения
                image_annotations[defect_name] = bbox_rect

            # Добавляем аннотации этого изображения в общий словарь
            annotations[image_id] = image_annotations

        # Для пустых изображений не добавляем запись в annotations.json
        # (согласно вашему примеру, там только изображения с дефектами)

        progress_bar.update(1)

    progress_bar.close()

    # Сохраняем аннотации в JSON файл
    annotations_file = os.path.join(directory, "annotations.json")
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    # Создаем отдельный файл с именами изображений
    image_list = []
    for i in range(num_images):
        image_id = f"Н{i+1}"
        image_filename = f"{image_id}.png"
        if os.path.exists(os.path.join(images_dir, image_filename)):
            image_list.append(image_id)

    with open(os.path.join(directory, "image_list.txt"), 'w', encoding='utf-8') as f:
        for img_id in image_list:
            f.write(f"{img_id}\n")

    print(f"\nДатасет успешно сгенерирован в {directory}")


def test_single_generation():
    """
    Тестирует генерацию одного изображения каждого типа с отображением bbox
    """
    print("Тестирование генерации с визуализацией bbox...")

    # Создаем тестовую директорию
    os.makedirs("test_output", exist_ok=True)

    test_cases = [
        ("crack_single", lambda: make_a_crack("single")),
        ("crack_tree", lambda: make_a_crack("tree")),
        ("crack_shattered", lambda: make_a_crack("shattered")),
        ("crack_transverse", lambda: make_a_crack("transverse")),
        ("incomplete_fusion", make_incomplete_fusion),
        ("single_pore", lambda: make_pore(num_pores=1)),
        ("cluster_pores", lambda: make_pore(num_pores=32)),
        ("empty", lambda: (make_empty_seam(), {}))
    ]

    for name, generator_func in test_cases:
        print(f"\nГенерация: {name}")

        if name == "empty":
            image = generator_func()
            bbox_dict = {}
        else:
            image, bbox_dict = generator_func()

        # Создаем копию с bbox
        if bbox_dict:
            image_with_bbox = image.copy()
            draw = ImageDraw.Draw(image_with_bbox)

            for defect_id, bbox_data in bbox_dict.items():
                if isinstance(bbox_data[0], (list, tuple)):
                    # Полигон
                    draw.polygon(bbox_data, outline="red", width=2)
                elif len(bbox_data) == 2:
                    # Точка
                    x, y = bbox_data
                    draw.ellipse([x-5, y-5, x+5, y+5], outline="green", width=2)
                elif len(bbox_data) == 4:
                    # Прямоугольник
                    draw.rectangle(bbox_data, outline="blue", width=2)

            # Сохраняем оба изображения
            image.save(f"test_output/{name}_original.png")
            image_with_bbox.save(f"test_output/{name}_with_bbox.png")
            print(f"  Сохранено: test_output/{name}_with_bbox.png")
            print(f"  Количество объектов: {len(bbox_dict)}")
        else:
            image.save(f"test_output/{name}_original.png")
            print(f"  Сохранено: test_output/{name}_original.png (пустое)")

    print("\nТестирование завершено! Все изображения сохранены в папке 'test_output'")

def quick_generation():
    """
    Быстрая генерация небольшого датасета для тестирования
    """
    print("Быстрая генерация тестового датасета...")
    generate_dataset("test_dataset", num_images=20)
    print("\nТестовый датасет создан в папке 'test_dataset'")
    print("Для просмотра аннотаций запустите: python test_dataset/view_annotations.py")

if __name__ == "__main__":
    # Меню выбора действий
    print("Генератор синтетического датасета дефектов сварных швов")
    print("=" * 50)
    generate_dataset("C:/PycharmProjects/XVL/data/training", num_images=100)