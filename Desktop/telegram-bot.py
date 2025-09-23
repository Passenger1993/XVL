import telebot
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import cv2

"========================================================================================"
def draw_defects(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    # Копирование изображения для последующего рисования на нем
    output_image = image.copy()

    # Преобразование изображения в оттенки серого
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Бинаризация изображения (преобразование в черно-белое)
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Поиск контуров на бинаризованном изображении
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Создание списка для хранения найденных дефектов
    defects = []

    # Анализ контуров и определение типа дефекта для каждого контура
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Определение ширины дефекта
        width = w // 10
        # Определение высоты дефекта
        height = h // 10
        # Создание ограничивающего прямоугольника для дефекта
        bounding_rect = (x - width, y - height, w + 2 * width, h + 2 * height)
        # Вырезание дефекта из исходного изображения
        defect = image[y - height : y + h + height, x - width : x + w + width]
        # Определение типа дефекта на основе его формы
        if w > h:
            if w > 3 * h:
                defect_type = "непровар"
            else:
                defect_type = "трещина"
        else:
            if w > h:
                defect_type = "одиночное включение"
            else:
                defect_type = "скопление пор"
        # Добавление информации о дефекте в список
        defects.append((defect_type, bounding_rect))

    # Преобразование numpy-массива изображения в объект Image
    output_image = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(output_image)
    font = ImageFont.truetype("Arial.ttf", 12)

    # Рисование ограничивающих прямоугольников и подписей на изображении
    for defect_type, bounding_rect in defects:
        x, y, w, h = bounding_rect
        draw.rectangle([(x, y), (x + w, y + h)], outline=(0, 255, 0), width=2)
        draw.text((x, y - 10), defect_type, font=font, fill=(0, 255, 0))

    return output_image

"========================================================================================"


bot = telebot.TeleBot('7125279638:AAG1J6bMoDdMGPrj-pERpP87MtuA6HOlowk')

@bot.message_handler(commands=["start"])
def get_text_messages(message):
    bot.send_message(message.chat.id, "Приветствую.Я - телеграм-бот, реализуюший функционал программы X-Ray Vision Lab."
                                      "Я предназначен для анализа изображений сварных швов на наличие дефектов. Пришлите мне "
                                      "изображение в формате JPG/PNG, и я отображу на нём повреждённые области")

@bot.message_handler(content_types=['photo'])
def receive_image(message):
    try:
        file_info = bot.get_file(message.photo[-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)
        photo_path = 'defect.jpg'

        with open(photo_path, 'wb') as new_file:
            new_file.write(downloaded_file)

        if os.path.exists(photo_path):
            # Обработка изображения
            result_image = draw_defects(photo_path)

            # Сохранение обработанного изображения
            result_filename = 'result.jpg'
            result_image.save(result_filename)

            # Отправка обработанного изображения
            with open(result_filename, 'rb') as result_file:
                bot.send_photo(message.chat.id, result_file)

            # Удаление временных файлов
            os.remove(photo_path)
            os.remove(result_filename)
        else:
            bot.reply_to(message, 'Файл с изображением не найден.')

    except Exception as e:
        bot.reply_to(message, f'Произошла ошибка: {e}')


bot.polling(none_stop=True)