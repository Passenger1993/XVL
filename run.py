#!/usr/bin/env python3
"""
🎯 XVL (X-Ray Vision Lab) - Production Inference Pipeline
Главная точка входа для использования предобученной модели.

Основные команды:
  python run.py predict --image path/to/image.jpg        # Предсказание на одном изображении
  python run.py predict --dir path/to/images/           # Пакетная обработка
  python run.py demo --count 5                         # Демо с генерацией данных
  python run.py web                                     # Запуск веб-интерфейса (опционально)
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

# Настройка пути для импорта модулей проекта
sys.path.insert(0, str(Path(__file__).parent))

# Импорт наших модулей
from src.utils.logging_config import setup_logging
from src.inference.predictor import XVLPredictor, PredictionResult
from src.generators import DemoDataGenerator
from src.utils.exсeptions import XVLBaseException, ModelLoadError, DataGenerationError

# Настройка логирования
logger = logging.getLogger(__name__)

class XVLCLI:
    """Клиент для командной строки XVL"""

    def __init__(self):
        self.predictor = None
        self.log_dir = None

    def setup_environment(self, verbose: bool = False):
        """Настройка окружения и логирования"""
        log_level = logging.DEBUG if verbose else logging.INFO
        self.log_dir = setup_logging(
            experiment_name="inference_run",
            log_level=log_level
        )
        logger.info("=" * 60)
        logger.info("XVL Inference Pipeline initialized")
        logger.info("=" * 60)

    def load_model(self, model_repo: str = None, device: str = None):
        """Загрузка модели с Hugging Face Hub"""
        try:
            logger.info(f"Loading model from Hugging Face Hub...")

            # Если репозиторий не указан, используем дефолтный
            if not model_repo:
                model_repo = "yourusername/xvl-weld-defect-detection"  # ← ЗАМЕНИТЕ на ваш репозиторий
                logger.info(f"Using default model repository: {model_repo}")

            self.predictor = XVLPredictor(
                model_repo=model_repo,
                device=device
            )

            logger.info("✅ Model loaded successfully")
            logger.info(f"   Device: {self.predictor.device}")
            logger.info(f"   Classes: {list(self.predictor.class_names.values())}")

        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise ModelLoadError(
                f"Cannot load model from {model_repo}",
                details={"error": str(e), "repo": model_repo}
            )

    def predict_image(self, image_path: str, output_dir: Optional[str] = None) -> PredictionResult:
        """Предсказание для одного изображения"""
        if not self.predictor:
            self.load_model()

        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        logger.info(f"Processing image: {image_path.name}")

        # Выполняем предсказание
        result = self.predictor.predict(str(image_path))

        # Сохраняем результаты если нужно
        if output_dir:
            save_path = self.predictor.save_result(result, output_dir)
            logger.info(f"Results saved to: {save_path}")

        # Логируем статистику
        defects_found = len(result.detections)
        logger.info(f"Found {defects_found} defect(s)")

        if defects_found > 0:
            for det in result.detections:
                logger.info(f"   - {det['class']}: {det['confidence']:.1%} "
                          f"at [{det['bbox'][0]:.0f}, {det['bbox'][1]:.0f}, "
                          f"{det['bbox'][2]:.0f}, {det['bbox'][3]:.0f}]")

        return result

    def predict_directory(self, dir_path: str, output_dir: str,
                         batch_size: int = 8) -> List[PredictionResult]:
        """Пакетная обработка изображений в директории"""
        if not self.predictor:
            self.load_model()

        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

        # Ищем изображения
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in dir_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        if not image_files:
            raise DataGenerationError(f"No images found in {dir_path}")

        logger.info(f"Found {len(image_files)} images in {dir_path}")

        # Создаем выходную директорию
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        results = []
        for i, img_path in enumerate(image_files):
            try:
                logger.info(f"[{i+1}/{len(image_files)}] Processing {img_path.name}")
                result = self.predict_image(str(img_path), str(output_path))
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {img_path.name}: {str(e)}")
                continue

        # Генерируем сводный отчет
        self._generate_summary_report(results, output_path)

        return results

    def run_demo(self, count: int = 5, output_dir: str = "demo_results"):
        """Запуск демо с генерацией синтетических данных"""
        logger.info(f"Starting demo mode with {count} generated images")

        # Генератор демо-данных
        generator = DemoDataGenerator()

        # Создаем выходную директорию
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Загружаем модель если ещё не загружена
        if not self.predictor:
            self.load_model()

        for i in range(count):
            try:
                logger.info(f"Generating demo image {i+1}/{count}...")

                # Генерируем изображение и аннотации
                image, true_annotations = generator.generate()

                # Сохраняем исходное изображение
                input_path = output_path / f"demo_{i:03d}_input.jpg"
                image.save(input_path)

                # Делаем предсказание
                result = self.predictor.predict(str(input_path))

                # Сохраняем результат
                result_path = output_path / f"demo_{i:03d}_result.jpg"
                result.annotated_image.save(result_path)

                # Сохраняем метаданные
                metadata = {
                    "image": f"demo_{i:03d}_input.jpg",
                    "true_defects": len(true_annotations),
                    "detected_defects": len(result.detections),
                    "defects": [
                        {
                            "class": det["class"],
                            "confidence": float(det["confidence"]),
                            "bbox": [float(c) for c in det["bbox"]]
                        }
                        for det in result.detections
                    ]
                }

                # Можно сохранить в JSON
                import json
                metadata_path = output_path / f"demo_{i:03d}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)

                logger.info(f"  → True defects: {len(true_annotations)}, "
                          f"Detected: {len(result.detections)}")

            except Exception as e:
                logger.error(f"Failed to generate demo {i+1}: {str(e)}")
                continue

        logger.info(f"✅ Demo completed. Results saved to: {output_path}")
        logger.info(f"   To view: open {output_path}/demo_*.jpg")

    def _generate_summary_report(self, results: List[PredictionResult], output_dir: Path):
        """Генерация сводного отчета по пакетной обработке"""
        if not results:
            return

        total_images = len(results)
        total_defects = sum(len(r.detections) for r in results)
        defect_by_class = {}

        for result in results:
            for det in result.detections:
                class_name = det["class"]
                defect_by_class[class_name] = defect_by_class.get(class_name, 0) + 1

        # Создаем текстовый отчет
        report_path = output_dir / "inference_summary.txt"
        with open(report_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("XVL Inference Summary Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Total images processed: {total_images}\n")
            f.write(f"Total defects detected: {total_defects}\n")
            f.write(f"Average defects per image: {total_defects/total_images:.2f}\n\n")
            f.write("Defects by class:\n")
            for class_name, count in defect_by_class.items():
                percentage = (count / total_defects * 100) if total_defects > 0 else 0
                f.write(f"  - {class_name}: {count} ({percentage:.1f}%)\n")

        logger.info(f"Summary report saved to: {report_path}")

    def run_web_interface(self, host: str = "127.0.0.1", port: int = 7860):
        """Запуск веб-интерфейса (опционально)"""
        try:
            # Проверяем установку gradio
            import importlib
            spec = importlib.util.find_spec("gradio")
            if spec is None:
                logger.warning("Gradio not installed. Install with: pip install gradio")
                logger.info("You can also use our Hugging Face Space:")
                logger.info("https://huggingface.co/spaces/yourusername/xvl-demo")
                return

            from src.web.app import create_app
            app = create_app(self.predictor)

            logger.info(f"Starting web interface at http://{host}:{port}")
            logger.info("Press Ctrl+C to stop")

            # Запускаем приложение
            app.launch(
                server_name=host,
                server_port=port,
                share=False  # Для локального использования
            )

        except ImportError:
            logger.error("Web interface dependencies not installed.")
            logger.info("Install with: pip install gradio pillow")
        except Exception as e:
            logger.error(f"Failed to start web interface: {str(e)}")

def main():
    """Основная функция обработки команд"""
    parser = argparse.ArgumentParser(
        description="XVL: X-Ray Vision Lab - Defect Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Предсказание для одного изображения
  python run.py predict --image examples/test_defect.jpg --output results/
  
  # Пакетная обработка папки
  python run.py predict --dir data/scans/ --output batch_results/
  
  # Демо-режим (генерация 5 тестовых изображений)
  python run.py demo --count 5
  
  # Запуск веб-интерфейса
  python run.py web --port 8080
  
  # Подробный вывод
  python run.py predict --image test.jpg --verbose
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Команды")

    # Парсер для predict
    predict_parser = subparsers.add_parser("predict", help="Предсказание дефектов")
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument("--image", help="Путь к изображению")
    predict_group.add_argument("--dir", help="Папка с изображениями")
    predict_parser.add_argument("--output", default="./results",
                               help="Папка для результатов (по умолчанию: ./results)")
    predict_parser.add_argument("--model", help="HF репозиторий модели (опционально)")
    predict_parser.add_argument("--device", choices=["cpu", "cuda", "auto"],
                               default="auto", help="Устройство для инференса")

    # Парсер для demo
    demo_parser = subparsers.add_parser("demo", help="Демо-режим с генерацией данных")
    demo_parser.add_argument("--count", type=int, default=5,
                            help="Количество генерируемых изображений")
    demo_parser.add_argument("--output", default="./demo_results",
                            help="Папка для результатов демо")

    # Парсер для web
    web_parser = subparsers.add_parser("web", help="Запуск веб-интерфейса")
    web_parser.add_argument("--host", default="127.0.0.1", help="Хост для веб-сервера")
    web_parser.add_argument("--port", type=int, default=7860, help="Порт для веб-сервера")

    # Общие аргументы
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Подробный вывод (debug режим)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    try:
        # Инициализируем CLI
        cli = XVLCLI()
        cli.setup_environment(verbose=args.verbose)

        # Обрабатываем команды
        if args.command == "predict":
            if args.image:
                cli.load_model(args.model, args.device)
                cli.predict_image(args.image, args.output)
            elif args.dir:
                cli.load_model(args.model, args.device)
                cli.predict_directory(args.dir, args.output)

        elif args.command == "demo":
            cli.run_demo(count=args.count, output_dir=args.output)

        elif args.command == "web":
            cli.load_model()  # Загружаем модель для веб-интерфейса
            cli.run_web_interface(host=args.host, port=args.port)

        logger.info("=" * 60)
        logger.info("✅ Operation completed successfully")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Process interrupted by user")
        return 130
    except FileNotFoundError as e:
        logger.error(f"❌ File error: {e}")
        return 1
    except ModelLoadError as e:
        logger.error(f"❌ Model loading failed: {e.message}")
        if e.details:
            logger.debug(f"Details: {e.details}")
        return 1
    except Exception as e:
        logger.error(f"❌ Unexpected error: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())