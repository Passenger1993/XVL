#!/usr/bin/python3

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.font as tkFont
from tkinter import ttk
from tkinter import filedialog
import tkinter.messagebox as mb
#import customtkinter
import numpy as np
import time
import os

from tensorflow import keras
#from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from PIL import Image, ImageTk


class XrayApp:
    def __init__(self, master = None):
        self.master = master
        self.master.title("X-Ray Vision Lab")
        self.app_resize(600, 350)
        self.defects = {
            0: "Непровар", 
            1: "Трещина", 
            2: "Одиночное включение", 
            3: "Скопление пор", 
            4: "Без дефекта"
        }

        # Disable resizing window due to fixed grids

        #self.master.resizable(width = False, height = False)

        self.draw()


    def draw(self) -> None:
        self.master.columnconfigure(0, weight = 1)

        self.main_frame = ttk.Frame(self.master)
        self.main_frame.pack(side = tk.TOP, fill = tk.BOTH, expand = False)

        self.main_frame.columnconfigure(0, weight = 1)
        self.main_frame.rowconfigure(0, weight = 1)
        self.main_frame.rowconfigure(1, weight = 1)
        self.main_frame.rowconfigure(2, weight = 1)

        self.main_menu = tk.Menu(self.master) 
        self.master.config(menu = self.main_menu) 
 
        self.file_menu = tk.Menu(self.main_menu, tearoff = 0)
        self.file_menu.add_command(label = "Открыть...", command = self.open_file)

        self.file_menu.add_separator()

        # Changing "state" if image is active, it will be more logical (normal, active, disabled)

        # Если разобраться как рисовать боксы на месте дефекта ->
        # https://pyimagesearch.com/2020/10/05/object-detection-bounding-box-regression-with-keras-tensorflow-and-deep-learning/
        # https://www.tensorflow.org/api_docs/python/tf/image/draw_bounding_boxes

        #self.file_menu.add_command(label = "Сохранить...")
        #self.file_menu.add_command(label = "Сохранить как...")

        self.file_menu.add_command(label = "Закрыть", command = self.close_file)
        self.file_menu.add_separator()

        # Таб с настройками, выбором модели

        #self.file_menu.add_command(label = "Настройки")
        #self.file_menu.add_separator()

        self.file_menu.add_command(label = "Выход", command = self.master.destroy)
 
        self.help_menu = tk.Menu(self.main_menu, tearoff = 0)
        self.help_menu.add_command(label = "Помощь", command = self.draw_help)
        self.help_menu.add_command(label = "О программе", command = self.draw_about)

        self.main_menu.add_cascade(label = "Файл", menu = self.file_menu)
        self.main_menu.add_command(label = "Старт", command = self.start_recognize)
        self.main_menu.add_cascade(label = "Справка", menu = self.help_menu)

        self.title_label = ttk.Label(self.main_frame)
        self.title_label.configure(text = 'X-Ray Vision Lab')
        self.title_label.config(font = ('Helvetica bold', 18))
        self.title_label.grid(column = 0, row = 0, pady = (15, 10), sticky = tk.NS)

        '''
        self.image_frame = ttk.Frame(self.main_frame)
        self.image_frame.grid(column = 0, row = 1, sticky = tk.NSEW)
        self.image_frame.grid_rowconfigure(0, weight = 1)
        self.image_frame.grid_columnconfigure(0, weight = 1)
        '''

        self.image_label = ttk.Label(self.main_frame)
        self.image_label.grid(column = 0, row = 1, pady = 5, sticky = tk.NS)

        self.title_file_name = ttk.Label(self.main_frame)
        self.title_file_name.configure(text = '')
        self.title_file_name.config(font = ('Helvetica bold', 14))
        self.title_file_name.grid(column = 0, row = 2, pady = (10, 15), sticky = tk.NS)


    def draw_help(self) -> None:
        newWindow = tk.Toplevel(self.master)
        newWindow.title("Помощь")
        newWindow.geometry("350x150")

        label = ttk.Label(newWindow, text = "© X-Ray Vision Lab. \nAll rights reserved.")
        label.pack(anchor = tk.CENTER, expand = 1)


    def draw_about(self) -> None:
        newWindow = tk.Toplevel(self.master)
        newWindow.title("О программе")
        newWindow.geometry("950x200")

        label = ttk.Label(newWindow, text = "Добро пожаловать в X-Ray Vision Lab! Это программу сверстали студенты СамГТУ из опорного вуза Самары.\n"
                          "Данная программа предназначена для сканирования\n"
                          "поверхностей заготовок на наличие различных видов дефектов, если таковые конечно имеются. Для начала работы \n"
                          "вам следует подать на вход изображение производного разрешения, но желательно чтобы оно не было слишком широким \n"
                          "или узким. На выходе вы получите результат сканирования. \n"
                          "В общем - желаем удачи в освоении нового инструмента!")
        label.config(font = ('Helvetica', 12))
        label.pack(anchor = tk.CENTER, expand = 1)


    def app_resize(self, width, height) -> None:

        if self.master.winfo_height() != height or self.master.winfo_width() != width:

            screenwidth = self.master.winfo_screenwidth()
            screenheight = self.master.winfo_screenheight()
            
            self.master.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))


    def open_file(self) -> None:
        try:
            filetypes = (('Файлы .png', '*.jpg'), ('Файлы .jpg', '*.png'), ('Все файлы', '*.*'),)

            self.filename = filedialog.askopenfilename(title = 'Открыть файл...', filetypes = filetypes)

            if self.filename == '':
                return

            if 'image' in dir(self):
                if self.current_file == self.filename:
                    return

            self.image = Image.open(self.filename)
            self.image = self.image.resize((200, 200))
            self.image = ImageTk.PhotoImage(self.image)

            self.current_file = self.filename
            self.title_file_name.config(text = 'Дефект: не определен')
            self.image_label.config(image = self.image)
        except Exception as e:
                mb.showerror(title = 'Исключение!', message = '{}'.format(e).title())
                self.close_file()


    def close_file(self) -> None:
        if 'image' in dir(self):
            del self.image

            self.image_label.config(image = None)
            self.current_file = None
            self.title_file_name.config(text = '')


    def tensor_image(self, path) -> None:
        try:
            with Image.open(path) as img:
                img = img.resize((64, 64))
                data_list = list(img.getdata())

                for num in range(len(data_list)):
                    data_list[num] = data_list[num][0] / 255

                tensor = np.array(data_list).reshape((64, 64))
                return tensor
        except Exception as e:
            mb.showerror(title = 'Исключение!', message = '{}'.format(e).title())


    def start_recognize(self) -> None:
        try:
            if 'image' not in dir(self):
                mb.showerror(title = 'Ошибка!', message = 'Выберите изображение!')
                return

            model = keras.models.load_model('v2.h5', compile = False)
            res = model.predict(np.expand_dims(self.tensor_image(self.filename), axis = 0), verbose = 0)
        
            if int(np.argmax(res)) != 4:
                mb.showwarning(title = 'Внимание!', message = "Обнаружен дефект: {}.".format(self.defects[int(np.argmax(res))]))
                self.title_file_name.config(text = 'Дефект: {}'.format(self.defects[int(np.argmax(res))]))
            else:
                mb.showinfo(title = 'Внимание!', message = 'Дефектов не найдено!')
        except Exception as e:
            mb.showerror(title = 'Исключение!', message = '{}'.format(e).title())


if __name__ == "__main__":
    root = tk.Tk()
    app = XrayApp(root)
    root.mainloop()


    #
    #  cd C:\Users\amade\AppData\Roaming\Python\Python37\Scripts
    #  pyinstaller --onefile G:\xray\xray\main.py
    #
