import json
import os
import sys
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import random

def load_annotations(annotations_path):
    """Загружает аннотации из JSON файла"""
    if not os.path.exists(annotations_path):
        print(f"Файл аннотаций не найден: {annotations_path}")
        return None

    with open(annotations_path, 'r', encoding='utf-8') as f:
        annotations = json.load(f)

    return annotations

def display_image_with_bboxes(image_path, annotations, image_id):
    """Отображает изображение с bbox"""
    if not os.path.exists(image_path):
        print(f"Изображение не найдено: {image_path}")
        return

    # Загружаем изображение
    image = Image.open(image_path).convert('L')  # Одноканальное
    image_rgb = Image.new('RGB', image.size)
    image_rgb.paste(image)  # Конвертируем в RGB для цветных bbox

    draw = ImageDraw.Draw(image_rgb)

    # Получаем аннотации для этого изображения
    if image_id not in annotations:
        print(f"Аннотации для изображения {image_id} не найдены")
        # Показываем изображение без bbox
        plt.figure(figsize=(10, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"Изображение {image_id} (без аннотаций)")
        plt.axis('off')
        plt.show()
        return

    image_annotations = annotations[image_id]

    if not image_annotations:
        print(f"Для изображения {image_id} нет аннотаций (пустое изображение)")
        plt.figure(figsize=(10, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"Изображение {image_id} (пустое, без дефектов)")
        plt.axis('off')
        plt.show()
        return

    print(f"Изображение: {image_id}.png")
    print(f"Размер: {image.size}")
    print(f"Количество дефектов: {len(image_annotations)}")
    print("-" * 50)

    # Цвета для разных типов дефектов
    colors = {
        "Трещина": (255, 0, 0),      # Красный
        "Непровар": (0, 0, 255),     # Синий
        "Скопление_пор": (0, 255, 0), # Зеленый
        "Одиночное_включение": (255, 165, 0),  # Оранжевый
    }

    # Рисуем bbox и собираем информацию
    defect_info = []
    for defect_name, bbox in image_annotations.items():
        # Определяем тип дефекта для выбора цвета
        defect_type = defect_name.split("_№")[0] if "_№" in defect_name else defect_name

        # Выбираем цвет
        color = colors.get(defect_type, (255, 255, 0))  # Желтый по умолчанию

        # Рисуем прямоугольник
        draw.rectangle(bbox, outline=color, width=2)

        # Добавляем текстовую метку
        x1, y1, x2, y2 = bbox
        # Упрощенная метка без фона
        draw.text((x1, y1 - 15), defect_name, fill=color)

        # Сохраняем информацию для вывода в консоль
        defect_info.append(f"  - {defect_name}: {bbox} (цвет: {defect_type})")

    # Выводим информацию о дефектах
    for info in defect_info:
        print(info)

    # Отображаем изображение с помощью matplotlib
    plt.figure(figsize=(12, 8))

    # Показываем оригинальное изображение
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Оригинал: {image_id}.png")
    plt.axis('off')

    # Показываем изображение с bbox
    plt.subplot(1, 2, 2)
    plt.imshow(image_rgb)
    plt.title(f"С разметкой: {image_id}.png\n{len(image_annotations)} дефектов")
    plt.axis('off')

    # Легенда для цветов
    legend_text = "Легенда:\n"
    for defect_type, color in colors.items():
        rgb_color = tuple(c/255 for c in color)  # Нормализуем для matplotlib
        legend_text += f"  {defect_type}: {color}\n"

    plt.figtext(0.02, 0.02, legend_text, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

    plt.tight_layout()
    plt.show()

def main():
    """Основная функция"""
    if len(sys.argv) < 2:
        # Интерактивный режим
        print("=" * 60)
        print("ВИЗУАЛЬНАЯ ПРОВЕРКА РАЗМЕТКИ BBOX")
        print("=" * 60)

        # Запрашиваем путь к датасету
        default_dataset = "C:/PycharmProjects/XVL/src/generators/real_samples"
        dataset_dir = input(f"Путь к датасету [{default_dataset}]: ").strip()
        if not dataset_dir:
            dataset_dir = default_dataset

        # Определяем пути
        images_dir = os.path.join(dataset_dir)
        annotations_path = os.path.join(dataset_dir,"annotations.json")

        # Загружаем аннотации
        annotations = load_annotations(annotations_path)
        if annotations is None:
            return

        while True:
            print("\n" + "=" * 60)
            print("Доступные изображения с аннотациями:")
            print("-" * 50)

            # Показываем первые 20 изображений
            for i, image_id in enumerate(sorted(annotations.keys(), key=lambda x: int(x))):
                if i >= 20:
                    print(f"... и еще {len(annotations) - 20} изображений")
                    break
                num_defects = len(annotations[image_id])
                image_path = os.path.join(images_dir, f"{image_id}.png")

                if os.path.exists(image_path):
                    status = "✓"
                else:
                    status = "✗"

                print(f"  {image_id:5} | Дефектов: {num_defects:2} {status}")

            print("-" * 50)
            print("Команды:")
            print("  [номер] - показать изображение с этим ID")
            print("  r [номер] - показать случайное изображение")
            print("  q - выход")
            print("=" * 60)

            command = input("Введите команду: ").strip().lower()

            if command == 'q':
                print("Выход.")
                break

            elif command.startswith('r'):
                # Случайное изображение
                if len(command.split()) > 1:
                    try:
                        count = int(command.split()[1])
                        for _ in range(count):
                            image_id = random.choice(list(annotations.keys()))
                            image_path = os.path.join(images_dir, f"{image_id}.png")
                            display_image_with_bboxes(image_path, annotations, image_id)
                    except:
                        image_id = random.choice(list(annotations.keys()))
                        image_path = os.path.join(images_dir, f"{image_id}.png")
                        display_image_with_bboxes(image_path, annotations, image_id)
                else:
                    image_id = random.choice(list(annotations.keys()))
                    image_path = os.path.join(images_dir, f"{image_id}.png")
                    display_image_with_bboxes(image_path, annotations, image_id)

            else:
                # Проверяем, является ли ввод числом
                try:
                    image_id = command
                    image_path = os.path.join(images_dir, f"{image_id.upper()}.png")
                    display_image_with_bboxes(image_path, annotations, image_id)
                except Exception as e:
                    print(f"Ошибка: {e}")
                    print("Попробуйте снова.")

    else:
        # Режим командной строки
        if len(sys.argv) == 3:
            dataset_dir = sys.argv[1]
            image_id = sys.argv[2]
        elif len(sys.argv) == 2:
            dataset_dir = "synthetic_dataset"
            image_id = sys.argv[1]
        else:
            print("Использование:")
            print("  python markup_test.py [путь_к_датасету] ID_изображения")
            print("  python markup_test.py ID_изображения (использует synthetic_dataset)")
            print("  python markup_test.py (интерактивный режим)")
            return

        # Определяем пути
        images_dir = os.path.join(dataset_dir)
        annotations_path = os.path.join(dataset_dir, "annotations.json")

        # Загружаем аннотации
        annotations = load_annotations(annotations_path)
        if annotations is None:
            return

        image_path = os.path.join(images_dir, f"{image_id}.png")
        display_image_with_bboxes(image_path, annotations, image_id)

if __name__ == "__main__":
    main()