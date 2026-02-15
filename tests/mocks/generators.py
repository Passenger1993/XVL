"""
Моки для генераторов изображений
"""

import sys
import numpy as np
from PIL import Image

# Мокируем модули, которые могут отсутствовать
sys.modules['src.generators.defects.crack'] = type(sys)('mock_crack')
sys.modules['src.generators.defects.incomplete_fusion'] = type(sys)('mock_incomplete_fusion')
sys.modules['src.generators.defects.pore'] = type(sys)('mock_pore')
sys.modules['src.generators.defects.empty'] = type(sys)('mock_empty')
sys.modules['src.generators.utils.zip_packer'] = type(sys)('mock_zip_packer')

# Функции-заглушки
def make_a_crack(crack_type="single"):
    img = Image.new('L', (256, 256), color=128)
    bbox = {"crack_1": [50, 50, 100, 100]}
    return img, bbox

def make_incomplete_fusion():
    img = Image.new('L', (256, 256), color=128)
    bbox = {"fusion_1": [60, 60, 110, 110]}
    return img, bbox

def make_pore(num_pores=1):
    img = Image.new('L', (256, 256), color=128)
    if num_pores == 1:
        bbox = {"pore_1": [70, 70, 80, 80]}
    else:
        bbox = {f"pore_{i}": [60 + i*10, 60 + i*10, 70 + i*10, 70 + i*10]
               for i in range(num_pores)}
    return img, bbox

def make_empty_seam():
    return Image.new('L', (256, 256), color=128)

def create_annotated_zip(input_zip, output_zip, json_path, copy_original=True):
    """Мок создания архива"""
    return None