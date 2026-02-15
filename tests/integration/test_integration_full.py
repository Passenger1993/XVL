"""
Интеграционные тесты для полного цикла работы системы
"""

import pytest
import sys
import json
import zipfile
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

# Добавляем путь к основному проекту
sys.path.append(str(Path(__file__).parent.parent))

class TestFullIntegration:
    """Тесты полного цикла работы системы"""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_pipeline_from_generation_to_evaluation(self, tmp_path):
        """Полный тест цикла: генерация → оценка → упаковка"""
        # Шаг 1: Генерация датасета
        with patch('manager.make_a_crack') as mock_crack, \
             patch('manager.make_incomplete_fusion') as mock_fusion, \
             patch('manager.make_pore') as mock_pore, \
             patch('manager.make_empty_seam') as mock_empty, \
             patch('manager.create_annotated_zip') as mock_zip, \
             patch('manager.load_real_samples_without_padding') as mock_load:

            # Настраиваем моки
            test_image = Image.new('L', (256, 256), color=128)

            mock_crack.return_value = (test_image, {"crack_1": [50, 50, 100, 100]})
            mock_fusion.return_value = (test_image, {"fusion_1": [60, 60, 110, 110]})
            mock_pore.return_value = (test_image, {"pore_1": [70, 70, 80, 80]})
            mock_empty.return_value = test_image
            mock_zip.return_value = None
            mock_load.return_value = ([], {})

            from manager import save_dataset

            dataset_dir = tmp_path / "generated_dataset"
            save_dataset(
                directory=str(dataset_dir),
                num_images=20,
                original_step=0,
                min_blur=0,
                max_blur=0,
                create_zip=False
            )

            # Проверяем генерацию
            assert dataset_dir.exists()
            assert (dataset_dir / "annotations.json").exists()

            image_files = list(dataset_dir.glob("*.png"))
            assert len(image_files) == 20

        # Шаг 2: Оценка датасета
        from evaluator import batch_evaluate_with_report

        evaluation_dir = tmp_path / "evaluation"
        summary, total_score = batch_evaluate_with_report(
            str(dataset_dir),
            str(evaluation_dir),
            sample_size=20
        )

        # Проверяем оценку
        assert summary is not None
        assert 0 <= total_score <= 10
        assert (evaluation_dir / "simple_summary.json").exists()
        assert (evaluation_dir / "verbal_report.txt").exists()

        # Шаг 3: Создание размеченного архива
        # Создаем ZIP из сгенерированных изображений
        input_zip = tmp_path / "dataset.zip"
        with zipfile.ZipFile(input_zip, 'w') as zipf:
            for img_file in dataset_dir.glob("*.png"):
                zipf.write(img_file, img_file.name)

        from zip_packer import create_annotated_zip

        output_zip = tmp_path / "annotated_dataset.zip"
        create_annotated_zip(
            input_zip=str(input_zip),
            output_zip=str(output_zip),
            json_path=str(dataset_dir / "annotations.json"),
            copy_original=False
        )

        # Проверяем упаковку
        assert output_zip.exists()

        with zipfile.ZipFile(output_zip, 'r') as zipf:
            files = zipf.namelist()
            assert len(files) == 20  # Все изображения должны быть в архиве

        # Итоговая проверка: все этапы выполнены успешно
        print(f"\nПолный цикл выполнен успешно!")
        print(f"  Сгенерировано изображений: 20")
        print(f"  Оценка датасета: {total_score:.1f}/10")
        print(f"  Создан архив с разметкой: {output_zip}")

    @pytest.mark.integration
    def test_cross_module_interaction(self, tmp_path):
        """Тест взаимодействия между модулями"""
        # Создаем тестовые данные
        dataset_dir = tmp_path / "test_dataset"
        dataset_dir.mkdir()

        # Создаем 5 тестовых изображений
        for i in range(5):
            img = Image.new('L', (200, 200), color=128)
            img.save(dataset_dir / f"{i+1}.png")

        # Создаем аннотации
        annotations = {
            "1": {"Трещина_№1": [10, 10, 50, 50]},
            "2": {"Непровар_№1": [20, 20, 60, 60]},
            "3": {"Одиночное_включение_№1": [30, 30, 40, 40]},
            "4": {},  # Без дефектов
            "5": {"Скопление_пор_№1": [40, 40, 80, 80]},
        }

        with open(dataset_dir / "annotations.json", 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False)

        # Шаг 1: Используем evaluator для оценки
        from evaluator import batch_evaluate_simple

        eval_dir = tmp_path / "evaluation"
        summary = batch_evaluate_simple(
            str(dataset_dir),
            str(eval_dir),
            sample_size=10
        )

        assert summary is not None
        assert summary['total_images'] == 5

        # Шаг 2: Используем zip_packer для создания архива
        input_zip = tmp_path / "input.zip"
        with zipfile.ZipFile(input_zip, 'w') as zipf:
            for img_file in dataset_dir.glob("*.png"):
                zipf.write(img_file, img_file.name)

        from zip_packer import create_annotated_zip

        output_zip = tmp_path / "output.zip"
        create_annotated_zip(
            input_zip=str(input_zip),
            output_zip=str(output_zip),
            json_path=str(dataset_dir / "annotations.json"),
            copy_original=False
        )

        # Шаг 3: Проверяем, что аннотации из evaluator соответствуют исходным
        with open(dataset_dir / "annotations.json", 'r', encoding='utf-8') as f:
            original_annotations = json.load(f)

        # Аннотации в summary должны соответствовать
        for image_result in summary['detailed_results']:
            image_id = image_result['image_id']
            defect_count = image_result['defect_count']

            if image_id in original_annotations:
                assert defect_count == len(original_annotations[image_id])
            else:
                assert defect_count == 0