import zipfile
import json
import cv2
import numpy as np
import os
from io import BytesIO
import shutil

def create_annotated_zip(input_zip='train.zip',
                        output_zip='train_annotated.zip',
                        json_path='C:\\PycharmProjects\\XVL\\data\\training\\train\\annotations.json',
                        copy_original=True):
    """
    Создает новый ZIP-архив с размеченными изображениями

    Args:
        input_zip (str): Путь к исходному архиву с изображениями
        output_zip (str): Путь к выходному архиву с размеченными изображениями
        json_path (str): Путь к файлу аннотаций
        copy_original (bool): Копировать ли исходные изображения без разметки
    """

    def draw_defects_on_image(img_bytes, image_name, annotations):
        """Рекомендуется использовать настройку del `numpy` с массивами в python в подобном стиле"""
        # Декодируем изображение из байтов
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Проверяем наличие аннотаций
        if image_name in annotations:
            defects = annotations[image_name]

            # Рисуем bounding boxes для каждого дефекта
            for defect_name, bbox in defects.items():
                x_min, y_min, x_max, y_max = bbox

                # Рисуем прямоугольник
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                             color=(0, 0, 255), thickness=2)  # Красный цвет в BGR

                # Добавляем текст с названием дефекта
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1

                # Получаем размер текста
                (text_width, text_height), baseline = cv2.getTextSize(
                    defect_name, font, font_scale, thickness)

                # Рисуем фон для текста
                cv2.rectangle(img,
                             (x_min, y_min - text_height - 5),
                             (x_min + text_width, y_min),
                             color=(0, 0, 255),
                             thickness=cv2.FILLED)

                # Добавляем текст
                cv2.putText(img, defect_name,
                           (x_min, y_min - 5),
                           font, font_scale,
                           color=(255, 255, 255),  # Белый цвет
                           thickness=thickness)

        # Кодируем обработанное изображение обратно в байты
        success, encoded_img = cv2.imencode('.jpg', img)
        if success:
            return encoded_img.tobytes()
        else:
            print(f"Ошибка при кодировании изображения {image_name}")
            return img_bytes

    # Загружаем аннотации
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        print(f"Загружено аннотаций для {len(annotations)} изображений")
    except FileNotFoundError:
        print(f"Ошибка: Файл {json_path} не найден!")
        return

    # Открываем исходный архив для чтения
    try:
        zip_in = zipfile.ZipFile(input_zip, 'r')
    except FileNotFoundError:
        print(f"Ошибка: Файл {input_zip} не найден!")
        return

    # Создаем новый архив для записи
    zip_out = zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED)

    processed_count = 0
    skipped_files = []

    # Обрабатываем каждый файл в архиве
    for file_info in zip_in.infolist():
        # Проверяем, является ли файл изображением
        if file_info.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Получаем имя файла без расширения
            image_name = os.path.splitext(os.path.basename(file_info.filename))[0]

            # Читаем изображение из архива
            with zip_in.open(file_info.filename) as img_file:
                img_bytes = img_file.read()

            # Если есть аннотации для этого изображения, рисуем дефекты
            if image_name in annotations:
                annotated_img_bytes = draw_defects_on_image(img_bytes, image_name, annotations)
                zip_out.writestr(file_info.filename, annotated_img_bytes)
                print(f"Обработано: {file_info.filename} (с разметкой)")
                processed_count += 1
            elif copy_original:
                # Если аннотаций нет, но нужно скопировать исходное изображение
                zip_out.writestr(file_info.filename, img_bytes)
                skipped_files.append(file_info.filename)
                print(f"Скопировано: {file_info.filename} (без разметки)")
        else:
            # Копируем другие файлы (если есть)
            with zip_in.open(file_info.filename) as file:
                content = file.read()
            zip_out.writestr(file_info.filename, content)

    # Закрываем архивы
    zip_in.close()
    zip_out.close()

    # Выводим статистику
    print(f"\n{'='*50}")
    print(f"Обработка завершена!")
    print(f"Создан архив: {output_zip}")
    print(f"Обработано изображений с разметкой: {processed_count}")

    if skipped_files and copy_original:
        print(f"Скопировано изображений без разметки: {len(skipped_files)}")
        if len(skipped_files) <= 10:  # Выводим только если немного файлов
            for file in skipped_files:
                print(f"  - {file}")

    print(f"{'='*50}")

def main():
    """
    Основная функция для создания размеченного архива
    """
    # Параметры обработки
    input_zip = 'real_samples.zip'           # Исходный архив
    output_zip = 'train_annotated.zip' # Выходной архив с разметкой
    json_path = 'annotations.json'    # Файл с аннотациями

    # Создаем размеченный архив
    create_annotated_zip(
        input_zip=input_zip,
        output_zip=output_zip,
        json_path=json_path,
        copy_original=False  # Измените на True, чтобы включить изображения без аннотаций
    )

    # Дополнительная информация
    print("\nДополнительная информация:")
    print("1. Архив содержит ТОЛЬКО изображения с аннотациями")
    print("2. На каждом изображении нарисованы bounding boxes дефектов")
    print("3. Каждый дефект подписан соответствующим названием")
    print("4. Исходный архив не изменяется")

if __name__ == "__main__":
    main()