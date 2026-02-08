# file name: quality_evaluator_final.py
import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# ОСНОВНЫЕ ФУНКЦИИ ОЦЕНКИ
# ========================================================

def simple_evaluate_image(image_path):
    """
    Упрощенная оценка изображения без сложной визуализации
    """
    # Загрузка изображения
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # Базовые метрики
    results = {
        'brightness_mean': float(np.mean(image)),
        'brightness_std': float(np.std(image)),
        'contrast': float(np.std(image)),  # Контраст как стандартное отклонение
        'entropy': simple_entropy(image),
        'edge_density': simple_edge_density(image),
        'file_size_kb': os.path.getsize(image_path) / 1024
    }

    return results

def simple_entropy(image):
    """Простая энтропия"""
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))

def simple_edge_density(image):
    """Плотность краев"""
    edges = cv2.Canny(image, 100, 200)
    return float(np.sum(edges > 0) / edges.size)

def batch_evaluate_simple(directory, output_dir, sample_size):
    """
    Упрощенная пакетная оценка
    """
    import glob

    # Находим все изображения
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []

    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(directory, ext)))

    # Сортируем по номеру
    try:
        image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.basename(x))) or 0))
    except:
        image_files.sort()

    # Ограничиваем выборку
    image_files = image_files[:sample_size]

    if not image_files:
        print(f"Изображения не найдены в директории: {directory}")
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Загружаем аннотации
    annotations_path = os.path.join(directory, "annotations.json")
    if os.path.exists(annotations_path):
        with open(annotations_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
    else:
        print(f"Предупреждение: Файл аннотаций не найден: {annotations_path}")
        annotations = {}

    all_results = []
    defect_statistics = {
        'Трещина': 0,
        'Непровар': 0,
        'Одиночное_включение': 0,
        'Скопление_пор': 0,
        'empty': 0
    }

    print(f"\nОценка {len(image_files)} изображений...")
    print("-" * 70)

    successful = 0
    failed = 0

    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        img_id = os.path.splitext(filename)[0]

        print(f"Обработка {i}/{len(image_files)}: {filename}")

        try:
            # Простая оценка
            results = simple_evaluate_image(image_path)

            if results:
                # Собираем статистику по дефектам
                defects_info = []
                if img_id in annotations:
                    for defect_name, bbox in annotations[img_id].items():
                        if "Трещина" in defect_name:
                            defect_type = "Трещина"
                        elif "Непровар" in defect_name:
                            defect_type = "Непровар"
                        elif "Одиночное_включение" in defect_name:
                            defect_type = "Одиночное_включение"
                        elif "Скопление_пор" in defect_name:
                            defect_type = "Скопление_пор"
                        else:
                            defect_type = "empty"

                        defect_statistics[defect_type] += 1
                        defects_info.append({
                            'name': defect_name,
                            'type': defect_type,
                            'bbox': bbox
                        })

                # Оценка качества (простая эвристика)
                # Чем ближе к реальным значениям, тем лучше
                # Типичные значения для рентгеновских снимков:
                # Яркость: 100-150, Контраст: 20-50, Энтропия: 4-7

                # Нормализованные оценки (0-100%)
                brightness_score = 100 * (1 - abs(results['brightness_mean'] - 125) / 125)
                contrast_score = 100 * (1 - abs(results['contrast'] - 35) / 35)
                entropy_score = 100 * (1 - abs(results['entropy'] - 5.5) / 5.5)

                # Общая оценка
                overall_score = (brightness_score + contrast_score + entropy_score) / 3

                all_results.append({
                    'filename': filename,
                    'image_id': img_id,
                    'brightness': results['brightness_mean'],
                    'contrast': results['contrast'],
                    'entropy': results['entropy'],
                    'edge_density': results['edge_density'],
                    'file_size_kb': results['file_size_kb'],
                    'brightness_score': brightness_score,
                    'contrast_score': contrast_score,
                    'entropy_score': entropy_score,
                    'overall_score': overall_score,
                    'defects': defects_info,
                    'defect_count': len(defects_info)
                })

                successful += 1
                print(f"  Оценка: {overall_score:.1f}%, Дефектов: {len(defects_info)}")
            else:
                failed += 1
                print(f"  Ошибка: не удалось оценить изображение")

        except Exception as e:
            failed += 1
            print(f"  Ошибка: {e}")
            # Не печатаем полный traceback, чтобы не загромождать вывод

    # Сводный отчет
    if all_results:
        print(f"\nУспешно оценено: {successful}, Не удалось: {failed}")

        summary = {
            'total_images': len(all_results),
            'successful': successful,
            'failed': failed,
            'average_scores': {
                'overall': np.mean([r['overall_score'] for r in all_results]),
                'brightness': np.mean([r['brightness_score'] for r in all_results]),
                'contrast': np.mean([r['contrast_score'] for r in all_results]),
                'entropy': np.mean([r['entropy_score'] for r in all_results])
            },
            'average_metrics': {
                'brightness': np.mean([r['brightness'] for r in all_results]),
                'contrast': np.mean([r['contrast'] for r in all_results]),
                'entropy': np.mean([r['entropy'] for r in all_results]),
                'edge_density': np.mean([r['edge_density'] for r in all_results]),
                'file_size_kb': np.mean([r['file_size_kb'] for r in all_results])
            },
            'defect_statistics': defect_statistics,
            'images_with_defects': sum(1 for r in all_results if r['defect_count'] > 0),
            'average_defects_per_image': np.mean([r['defect_count'] for r in all_results]),
            'score_distribution': {
                'excellent': sum(1 for r in all_results if r['overall_score'] >= 80),
                'good': sum(1 for r in all_results if 60 <= r['overall_score'] < 80),
                'fair': sum(1 for r in all_results if 40 <= r['overall_score'] < 60),
                'poor': sum(1 for r in all_results if r['overall_score'] < 40)
            },
            'detailed_results': all_results
        }

        # Сохраняем JSON отчет
        summary_path = os.path.join(output_dir, "simple_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Создаем визуализацию
        try:
            create_simple_visualizations(summary, output_dir)
        except Exception as e:
            print(f"Не удалось создать визуализации: {e}")

        return summary
    else:
        print("Не удалось оценить ни одного изображения")
        return None

def create_simple_visualizations(summary, output_dir):
    """Создание простых визуализаций"""
    # 1. График распределения оценок
    plt.figure(figsize=(10, 6))
    scores = [r['overall_score'] for r in summary['detailed_results']]
    plt.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=summary['average_scores']['overall'], color='red', linestyle='--',
               label=f'Среднее: {summary["average_scores"]["overall"]:.1f}%')
    plt.xlabel('Общая оценка качества (%)')
    plt.ylabel('Количество изображений')
    plt.title('Распределение оценок качества')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "score_distribution.png"), dpi=150)
    plt.close()

    # 2. График метрик
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Яркость
    axes[0, 0].hist([r['brightness'] for r in summary['detailed_results']], bins=20, alpha=0.7)
    axes[0, 0].set_xlabel('Яркость')
    axes[0, 0].set_ylabel('Частота')
    axes[0, 0].set_title('Распределение яркости')
    axes[0, 0].grid(True, alpha=0.3)

    # Контраст
    axes[0, 1].hist([r['contrast'] for r in summary['detailed_results']], bins=20, alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Контраст')
    axes[0, 1].set_title('Распределение контраста')
    axes[0, 1].grid(True, alpha=0.3)

    # Энтропия
    axes[1, 0].hist([r['entropy'] for r in summary['detailed_results']], bins=20, alpha=0.7, color='green')
    axes[1, 0].set_xlabel('Энтропия')
    axes[1, 0].set_ylabel('Частота')
    axes[1, 0].set_title('Распределение энтропии')
    axes[1, 0].grid(True, alpha=0.3)

    # Дефекты
    defect_types = [k for k, v in summary['defect_statistics'].items() if v > 0]
    defect_counts = [summary['defect_statistics'][k] for k in defect_types]
    axes[1, 1].bar(defect_types, defect_counts, alpha=0.7, color='red')
    axes[1, 1].set_xlabel('Тип дефекта')
    axes[1, 1].set_ylabel('Количество')
    axes[1, 1].set_title('Распределение дефектов')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Статистика синтетических изображений', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_distribution.png"), dpi=150)
    plt.close()

    # 3. Круговая диаграмма качества
    plt.figure(figsize=(8, 8))
    labels = ['Отлично', 'Хорошо', 'Удовлетворительно', 'Плохо']
    sizes = [
        summary['score_distribution']['excellent'],
        summary['score_distribution']['good'],
        summary['score_distribution']['fair'],
        summary['score_distribution']['poor']
    ]
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#F44336']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Качество синтетических изображений')
    plt.savefig(os.path.join(output_dir, "quality_pie.png"), dpi=150)
    plt.close()

    print(f"Визуализации сохранены в {output_dir}")

# ========================================================
# ФУНКЦИИ ДЛЯ СЛОВЕСНОГО ОТЧЕТА
# ========================================================

def generate_verbal_report(summary, output_dir):
    """
    Генерирует словесный отчет с 10-балльной шкалой
    """

    # 1. ОЦЕНКА ТЕХНИЧЕСКОГО КАЧЕСТВА ИЗОБРАЖЕНИЙ (0-10 баллов)
    def score_technical_quality(metrics):
        """
        Оценка технического качества изображений по 5 критериям
        """
        scores = []

        # 1.1 Яркость (идеально 100-150 для рентгена)
        brightness = metrics['brightness']
        if 100 <= brightness <= 150:
            brightness_score = 10
        elif 80 <= brightness <= 170:
            brightness_score = 8
        elif 60 <= brightness <= 190:
            brightness_score = 6
        elif 40 <= brightness <= 210:
            brightness_score = 4
        else:
            brightness_score = 2

        scores.append(('Яркость', brightness_score, brightness, "100-150"))

        # 1.2 Контраст (идеально 20-50)
        contrast = metrics['contrast']
        if 20 <= contrast <= 50:
            contrast_score = 10
        elif 15 <= contrast <= 60:
            contrast_score = 8
        elif 10 <= contrast <= 70:
            contrast_score = 6
        elif 5 <= contrast <= 85:
            contrast_score = 4
        else:
            contrast_score = 2

        scores.append(('Контраст', contrast_score, contrast, "20-50"))

        # 1.3 Энтропия (идеально 5-7)
        entropy = metrics['entropy']
        if 5 <= entropy <= 7:
            entropy_score = 10
        elif 4 <= entropy <= 8:
            entropy_score = 8
        elif 3 <= entropy <= 9:
            entropy_score = 6
        elif 2 <= entropy <= 10:
            entropy_score = 4
        else:
            entropy_score = 2

        scores.append(('Информационная энтропия', entropy_score, f"{entropy:.2f}", "5-7"))

        # 1.4 Плотность краев (идеально 0.05-0.15)
        edge_density = metrics['edge_density']
        if 0.05 <= edge_density <= 0.15:
            edge_score = 10
        elif 0.03 <= edge_density <= 0.20:
            edge_score = 8
        elif 0.01 <= edge_density <= 0.25:
            edge_score = 6
        elif 0.005 <= edge_density <= 0.30:
            edge_score = 4
        else:
            edge_score = 2

        scores.append(('Плотность краев', edge_score, f"{edge_density:.3f}", "0.05-0.15"))

        # 1.5 Стабильность размеров файлов (малое СКО)
        file_sizes = [r['file_size_kb'] for r in summary['detailed_results']]
        file_std = np.std(file_sizes)
        file_mean = np.mean(file_sizes)

        # Коэффициент вариации
        cv = file_std / file_mean if file_mean > 0 else 0

        if cv < 0.1:
            file_score = 10
        elif cv < 0.2:
            file_score = 8
        elif cv < 0.3:
            file_score = 6
        elif cv < 0.5:
            file_score = 4
        else:
            file_score = 2

        scores.append(('Стабильность размеров файлов', file_score, f"СКО={file_std:.1f}КБ", "СКО<20%"))

        return scores

    # 2. ОЦЕНКА ДЕФЕКТОВ (0-10 баллов)
    def score_defect_quality(defect_stats, avg_defects):
        """
        Оценка качества и разнообразия дефектов
        """
        scores = []

        # 2.1 Разнообразие типов дефектов
        defect_types = [k for k, v in defect_stats.items() if v > 0 and k != 'empty']
        num_types = len(defect_types)

        if num_types >= 4:
            diversity_score = 10
        elif num_types == 3:
            diversity_score = 8
        elif num_types == 2:
            diversity_score = 6
        elif num_types == 1:
            diversity_score = 4
        else:
            diversity_score = 2

        scores.append(('Разнообразие типов дефектов', diversity_score, f"{num_types} типов", "≥3 типа"))

        # 2.2 Распределение дефектов по типам (чем равномернее, тем лучше)
        if num_types > 1:
            type_counts = [defect_stats[t] for t in defect_types]
            # Коэффициент вариации распределения
            cv_dist = np.std(type_counts) / np.mean(type_counts) if np.mean(type_counts) > 0 else 1

            if cv_dist < 0.5:
                distribution_score = 10
            elif cv_dist < 1.0:
                distribution_score = 8
            elif cv_dist < 1.5:
                distribution_score = 6
            elif cv_dist < 2.0:
                distribution_score = 4
            else:
                distribution_score = 2
        else:
            distribution_score = 3

        scores.append(('Равномерность распределения дефектов', distribution_score,
                      f"разброс {cv_dist:.1f}" if num_types > 1 else "только 1 тип", "разброс < 0.5"))

        # 2.3 Частота дефектов на изображение
        if 0.5 <= avg_defects <= 2.5:
            frequency_score = 10
        elif 0.2 <= avg_defects <= 3.0:
            frequency_score = 8
        elif 0.1 <= avg_defects <= 4.0:
            frequency_score = 6
        elif avg_defects <= 5.0:
            frequency_score = 4
        else:
            frequency_score = 2

        scores.append(('Частота дефектов на изображение', frequency_score, f"{avg_defects:.1f}", "0.5-2.5"))

        # 2.4 Соотношение изображений с дефектами и без
        total_images = summary['total_images']
        with_defects = summary['images_with_defects']
        defect_ratio = with_defects / total_images if total_images > 0 else 0

        if 0.4 <= defect_ratio <= 0.8:
            ratio_score = 10
        elif 0.3 <= defect_ratio <= 0.9:
            ratio_score = 8
        elif 0.2 <= defect_ratio <= 1.0:
            ratio_score = 6
        elif 0.1 <= defect_ratio <= 1.0:
            ratio_score = 4
        else:
            ratio_score = 2

        scores.append(('Соотношение с дефектами/без', ratio_score, f"{defect_ratio:.1%}", "40-80%"))

        return scores

    # 3. ОЦЕНКА ПРИГОДНОСТИ ВЫБОРКИ ДЛЯ ОБУЧЕНИЯ (0-10 баллов)
    def score_dataset_suitability(tech_scores, defect_scores, overall_score):
        """
        Общая оценка пригодности выборки для обучения моделей
        """
        scores = []

        # 3.1 Качество аннотаций
        total_defects = sum(summary['defect_statistics'].values())
        annotation_completeness = total_defects / summary['total_images'] if summary['total_images'] > 0 else 0

        if annotation_completeness >= 1.0:
            annotation_score = 10
        elif annotation_completeness >= 0.7:
            annotation_score = 8
        elif annotation_completeness >= 0.5:
            annotation_score = 6
        elif annotation_completeness >= 0.3:
            annotation_score = 4
        else:
            annotation_score = 2

        scores.append(('Качество аннотаций', annotation_score,
                      f"{annotation_completeness:.1f} деф/изобр", "≥1.0 деф/изобр"))

        # 3.2 Размер выборки
        sample_size = summary['total_images']

        if sample_size >= 500:
            size_score = 10
        elif sample_size >= 250:
            size_score = 8
        elif sample_size >= 100:
            size_score = 6
        elif sample_size >= 50:
            size_score = 4
        elif sample_size >= 20:
            size_score = 2
        else:
            size_score = 1

        scores.append(('Размер выборки', size_score, f"{sample_size} изображений", "≥100"))

        # 3.3 Сбалансированность оценок (малое СКО)
        quality_scores = [r['overall_score'] for r in summary['detailed_results']]
        quality_std = np.std(quality_scores)

        if quality_std <= 10:
            consistency_score = 10
        elif quality_std <= 15:
            consistency_score = 8
        elif quality_std <= 20:
            consistency_score = 6
        elif quality_std <= 25:
            consistency_score = 4
        else:
            consistency_score = 2

        scores.append(('Консистентность качества', consistency_score, f"СКО={quality_std:.1f}%", "СКО≤10%"))

        # 3.4 Общая оценка качества
        if overall_score >= 80:
            overall_quality_score = 10
        elif overall_score >= 70:
            overall_quality_score = 8
        elif overall_score >= 60:
            overall_quality_score = 6
        elif overall_score >= 50:
            overall_quality_score = 4
        else:
            overall_quality_score = 2

        scores.append(('Общее качество изображений', overall_quality_score,
                      f"{overall_score:.1f}%", "≥70%"))

        return scores

    # Генерируем оценки
    tech_scores = score_technical_quality(summary['average_metrics'])
    defect_scores = score_defect_quality(summary['defect_statistics'],
                                        summary['average_defects_per_image'])
    dataset_scores = score_dataset_suitability(tech_scores, defect_scores,
                                              summary['average_scores']['overall'])

    # Рассчитываем итоговые баллы
    tech_total = np.mean([score for _, score, _, _ in tech_scores])
    defect_total = np.mean([score for _, score, _, _ in defect_scores])
    dataset_total = np.mean([score for _, score, _, _ in dataset_scores])

    # Общий балл (взвешенный)
    total_score = 0.3 * tech_total + 0.3 * defect_total + 0.4 * dataset_total

    # Создаем текстовый отчет
    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("ЭКСПЕРТНЫЙ ОТЧЕТ О КАЧЕСТВЕ СИНТЕТИЧЕСКОЙ ВЫБОРКИ")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Общая информация
    report_lines.append(f"Общее количество изображений: {summary['total_images']}")
    report_lines.append(f"Изображений с дефектами: {summary['images_with_defects']} ({summary['images_with_defects']/summary['total_images']:.1%})")
    report_lines.append(f"Среднее количество дефектов на изображение: {summary['average_defects_per_image']:.1f}")
    report_lines.append("")

    # Раздел 1: Техническое качество
    report_lines.append("1. ТЕХНИЧЕСКОЕ КАЧЕСТВО ИЗОБРАЖЕНИЙ")
    report_lines.append("-" * 50)
    for name, score, value, ideal in tech_scores:
        stars = "★" * int(score) + "☆" * (10 - int(score))
        report_lines.append(f"  {name:30} {stars} ({score:.1f}/10)")
        report_lines.append(f"     Текущее значение: {value}, Идеал: {ideal}")
    report_lines.append(f"  ИТОГ: {tech_total:.1f}/10 баллов")
    report_lines.append("")

    # Раздел 2: Качество дефектов
    report_lines.append("2. КАЧЕСТВО И РАЗНООБРАЗИЕ ДЕФЕКТОВ")
    report_lines.append("-" * 50)
    for name, score, value, ideal in defect_scores:
        stars = "★" * int(score) + "☆" * (10 - int(score))
        report_lines.append(f"  {name:30} {stars} ({score:.1f}/10)")
        report_lines.append(f"     Текущее значение: {value}, Идеал: {ideal}")
    report_lines.append(f"  ИТОГ: {defect_total:.1f}/10 баллов")
    report_lines.append("")

    # Статистика по типам дефектов
    report_lines.append("  Статистика по типам дефектов:")
    for defect_type, count in summary['defect_statistics'].items():
        if count > 0:
            report_lines.append(f"    • {defect_type}: {count}")
    report_lines.append("")

    # Раздел 3: Пригодность для обучения
    report_lines.append("3. ПРИГОДНОСТЬ ВЫБОРКИ ДЛЯ ОБУЧЕНИЯ МОДЕЛЕЙ")
    report_lines.append("-" * 50)
    for name, score, value, ideal in dataset_scores:
        stars = "★" * int(score) + "☆" * (10 - int(score))
        report_lines.append(f"  {name:30} {stars} ({score:.1f}/10)")
        report_lines.append(f"     Текущее значение: {value}, Идеал: {ideal}")
    report_lines.append(f"  ИТОГ: {dataset_total:.1f}/10 баллов")
    report_lines.append("")

    # ИТОГОВАЯ ОЦЕНКА
    report_lines.append("=" * 80)
    report_lines.append("ИТОГОВАЯ ОЦЕНКА")
    report_lines.append("=" * 80)

    # Визуализация общего балла
    total_int = int(total_score)
    total_stars = "★" * total_int + "☆" * (10 - total_int)

    report_lines.append(f"ОБЩИЙ БАЛЛ: {total_stars} ({total_score:.1f}/10)")
    report_lines.append("")

    # Классификация качества
    if total_score >= 9.0:
        classification = "ОТЛИЧНО"
        recommendation = "Выборка полностью готова к использованию в производственных задачах."
        suitability = "Высокая пригодность"
    elif total_score >= 7.0:
        classification = "ХОРОШО"
        recommendation = "Выборка пригодна для обучения, возможны незначительные улучшения."
        suitability = "Хорошая пригодность"
    elif total_score >= 5.0:
        classification = "УДОВЛЕТВОРИТЕЛЬНО"
        recommendation = "Выборка требует доработки перед использованием в серьезных проектах."
        suitability = "Ограниченная пригодность"
    elif total_score >= 3.0:
        classification = "НИЗКОЕ КАЧЕСТВО"
        recommendation = "Требуется значительная доработка выборки."
        suitability = "Низкая пригодность"
    else:
        classification = "КРИТИЧЕСКО НИЗКОЕ КАЧЕСТВО"
        recommendation = "Выборка непригодна для обучения моделей без полной переработки."
        suitability = "Непригодна"

    report_lines.append(f"КЛАССИФИКАЦИЯ: {classification}")
    report_lines.append(f"ПРИГОДНОСТЬ ДЛЯ ОБУЧЕНИЯ: {suitability}")
    report_lines.append("")

    # Рекомендации
    report_lines.append("РЕКОМЕНДАЦИИ:")
    report_lines.append("-" * 30)
    report_lines.append(recommendation)
    report_lines.append("")

    # Конкретные рекомендации по улучшению
    recommendations = []

    # Анализируем слабые места
    weak_tech = [name for name, score, _, _ in tech_scores if score < 7]
    weak_defect = [name for name, score, _, _ in defect_scores if score < 7]
    weak_dataset = [name for name, score, _, _ in dataset_scores if score < 7]

    if weak_tech:
        recommendations.append(f"• Улучшить технические параметры: {', '.join(weak_tech)}")

    if weak_defect:
        recommendations.append(f"• Улучшить качество дефектов: {', '.join(weak_defect)}")

    if weak_dataset:
        recommendations.append(f"• Улучшить пригодность выборки: {', '.join(weak_dataset)}")

    # Проверка баланса дефектов
    defect_balance = summary['defect_statistics']
    if len([v for k, v in defect_balance.items() if v > 0 and k != 'empty']) == 1:
        recommendations.append("• Добавить больше разнообразия в типы дефектов")

    if summary['images_with_defects'] < summary['total_images'] * 0.3:
        recommendations.append("• Увеличить количество изображений с дефектами")

    if summary['total_images'] < 100:
        recommendations.append(f"• Увеличить размер выборки (сейчас {summary['total_images']}, рекомендуется ≥100)")

    if recommendations:
        report_lines.append("КОНКРЕТНЫЕ ШАГИ ПО УЛУЧШЕНИЮ:")
        for rec in recommendations:
            report_lines.append(f"  {rec}")

    report_lines.append("")
    report_lines.append("=" * 80)
    report_lines.append(f"Отчет сгенерирован: {summary.get('generation_date', '')}")
    report_lines.append("=" * 80)

    # Сохраняем отчет в файл
    report_path = os.path.join(output_dir, "verbal_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    # Создаем также краткий HTML отчет для удобства просмотра
    create_html_report(summary, total_score, classification, suitability,
                      tech_scores, defect_scores, dataset_scores, output_dir)

    return total_score, report_lines

def create_html_report(summary, total_score, classification, suitability,
                      tech_scores, defect_scores, dataset_scores, output_dir):
    """
    Создает HTML отчет с графиками
    """
    # Создаем график оценок с читаемыми подписями
    fig, ax = plt.subplots(figsize=(12, 8))

    categories = []
    scores = []

    # Техническое качество - короткие понятные названия
    for name, score, _, _ in tech_scores:
        # Создаем короткие, но понятные названия
        short_name = {
            'Яркость': 'Яркость',
            'Контраст': 'Контраст',
            'Информационная энтропия': 'Энтропия',
            'Плотность краев': 'Края',
            'Стабильность размеров файлов': 'Размеры файлов'
        }.get(name, name[:15])
        categories.append(f"Т: {short_name}")
        scores.append(score)

    # Качество дефектов
    for name, score, _, _ in defect_scores:
        short_name = {
            'Разнообразие типов дефектов': 'Разнообразие деф.',
            'Равномерность распределения дефектов': 'Распределение',
            'Частота дефектов на изображение': 'Частота деф.',
            'Соотношение с дефектами/без': 'Деф./Без деф.'
        }.get(name, name[:15])
        categories.append(f"Д: {short_name}")
        scores.append(score)

    # Пригодность для обучения
    for name, score, _, _ in dataset_scores:
        short_name = {
            'Качество аннотаций': 'Аннотации',
            'Размер выборки': 'Размер выборки',
            'Консистентность качества': 'Консистентность',
            'Общее качество изображений': 'Общее качество'
        }.get(name, name[:15])
        categories.append(f"О: {short_name}")
        scores.append(score)

    # Используем горизонтальные столбцы с читаемыми подписями
    colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
    bars = ax.barh(categories, scores, color=colors, height=0.6)
    ax.set_xlim(0, 10)
    ax.set_xlabel('Оценка (0-10 баллов)')
    ax.set_title('Детальная оценка качества выборки по критериям', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3, axis='x')

    # Добавляем значения на столбцы
    for bar, score in zip(bars, scores):
        width = bar.get_width()
        ax.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                f'{score:.1f}', ha='left', va='center', fontsize=9)

    # Увеличиваем отступы для подписей
    plt.tight_layout()

    # Сохраняем график
    chart_path = os.path.join(output_dir, "detailed_scores.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Создаем легенду для типов критериев
    fig_legend, ax_legend = plt.subplots(figsize=(8, 1))
    ax_legend.axis('off')

    legend_text = ("Т - Техническое качество изображений\n"
                   "Д - Качество и разнообразие дефектов\n"
                   "О - Пригодность для обучения моделей")

    ax_legend.text(0, 0.5, legend_text, fontsize=10,
                   verticalalignment='center', linespacing=1.5)

    legend_path = os.path.join(output_dir, "legend.png")
    plt.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Создаем HTML
    html_path = os.path.join(output_dir, "summary_report.html")

    html_content = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Отчет оценки качества синтетических изображений</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
            .header {{ background: #3498db; color: white; padding: 20px; border-radius: 5px; text-align: center; }}
            .total-score {{ font-size: 48px; font-weight: bold; color: #e74c3c; text-align: center; margin: 20px 0; }}
            .stars {{ color: #f1c40f; font-size: 24px; text-align: center; }}
            .section {{ margin: 30px 0; padding: 20px; background: #ecf0f1; border-radius: 5px; }}
            .metric {{ display: flex; justify-content: space-between; margin: 10px 0; padding: 10px; background: white; border-radius: 3px; }}
            .metric-name {{ flex: 2; }}
            .metric-score {{ flex: 1; text-align: right; }}
            .recommendations {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; }}
            .verdict {{ background: {'#d4edda' if total_score >= 7 else '#f8d7da'}; 
                       border: 1px solid {'#c3e6cb' if total_score >= 7 else '#f5c6cb'};
                       padding: 20px; border-radius: 5px; margin: 20px 0; }}
            .chart {{ text-align: center; margin: 20px 0; }}
            .chart-container {{ display: flex; justify-content: center; align-items: center; margin: 20px 0; }}
            .legend-box {{ background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #6c757d; margin: 15px 0; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>ЭКСПЕРТНЫЙ ОТЧЕТ О КАЧЕСТВЕ СИНТЕТИЧЕСКОЙ ВЫБОРКИ</h1>
            </div>
            
            <div style="text-align: center; margin: 20px 0;">
                <div class="total-score">{total_score:.1f}/10</div>
                <div class="stars">{"★" * int(total_score) + "☆" * (10 - int(total_score))}</div>
                <h2 style="color: {'#27ae60' if total_score >= 7 else '#e74c3c' if total_score >= 5 else '#c0392b'}">{classification}</h2>
                <p>Пригодность для обучения: <strong>{suitability}</strong></p>
            </div>
            
            <div class="verdict">
                <h3>ВЫВОД</h3>
                <p>Выборка {'полностью готова' if total_score >= 7 else 'требует доработки' if total_score >= 5 else 'не пригодна'} 
                для использования в обучении моделей компьютерного зрения. Общая оценка {total_score:.1f} баллов из 10.</p>
            </div>
            
            <div class="chart">
                <h3>📈 Детальная оценка по критериям</h3>
                <div class="legend-box">
                    <strong>Обозначения:</strong><br>
                    <strong>Т</strong> - Техническое качество изображений<br>
                    <strong>Д</strong> - Качество и разнообразие дефектов<br>
                    <strong>О</strong> - Пригодность для обучения моделей
                </div>
                <img src="detailed_scores.png" alt="Детальная оценка" style="max-width: 100%; border: 1px solid #ddd; padding: 10px; background: white;">
            </div>
            
            <div class="section">
                <h3>📊 Статистика выборки</h3>
                <div class="metric">
                    <span class="metric-name">Всего изображений:</span>
                    <span class="metric-score">{summary['total_images']}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Изображений с дефектами:</span>
                    <span class="metric-score">{summary['images_with_defects']} ({summary['images_with_defects']/summary['total_images']:.1%})</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Среднее дефектов на изображение:</span>
                    <span class="metric-score">{summary['average_defects_per_image']:.1f}</span>
                </div>
                <div class="metric">
                    <span class="metric-name">Средняя оценка качества:</span>
                    <span class="metric-score">{summary['average_scores']['overall']:.1f}%</span>
                </div>
            </div>
            
            <div class="section">
                <h3>🔧 Техническое качество изображений</h3>
                {''.join([f'<div class="metric"><span class="metric-name">{name}</span><span class="metric-score">{score:.1f}/10</span></div>' for name, score, _, _ in tech_scores])}
            </div>
            
            <div class="section">
                <h3>⚠️ Качество и разнообразие дефектов</h3>
                {''.join([f'<div class="metric"><span class="metric-name">{name}</span><span class="metric-score">{score:.1f}/10</span></div>' for name, score, _, _ in defect_scores])}
            </div>
            
            <div class="section">
                <h3>🎯 Пригодность для обучения моделей</h3>
                {''.join([f'<div class="metric"><span class="metric-name">{name}</span><span class="metric-score">{score:.1f}/10</span></div>' for name, score, _, _ in dataset_scores])}
            </div>
            
            <div class="recommendations">
                <h3>💡 Рекомендации</h3>
                <p>{'Выборка высокого качества. Можно использовать для обучения производственных моделей.' if total_score >= 7 
                    else 'Выборка удовлетворительного качества. Рекомендуется доработать перед использованием.' if total_score >= 5 
                    else 'Выборка низкого качества. Требуется значительная доработка.'}</p>
                
                <h4>Конкретные шаги по улучшению:</h4>
                <ul>
                    <li>Проверить баланс типов дефектов</li>
                    <li>Увеличить разнообразие фоновых текстур</li>
                    <li>Добавить больше вариативности в параметры генерации</li>
                    <li>Увеличить размер выборки до 100+ изображений</li>
                </ul>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 12px;">
                <p>Отчет сгенерирован автоматически на основе анализа {summary['total_images']} изображений</p>
                <p>Дата генерации: {summary.get('generation_date', '')}</p>
            </div>
        </div>
    </body>
    </html>
    '''

    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML отчет сохранен: {html_path}")

def batch_evaluate_with_report(directory, output_dir, sample_size):
    """
    Пакетная оценка с генерацией словесного отчета
    """
    # Выполняем пакетную оценку
    summary = batch_evaluate_simple(directory, output_dir, sample_size)

    if summary:
        # Добавляем дату генерации
        summary['generation_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Генерируем словесный отчет
        total_score, report_lines = generate_verbal_report(summary, output_dir)

        # Печатаем отчет в консоль
        print("\n" + "=" * 80)
        print("СЛОВЕСНЫЙ ОТЧЕТ")
        print("=" * 80)
        for line in report_lines[:30]:  # Выводим первые 30 строк
            print(line)

        print(f"\nПолный отчет сохранен в: {output_dir}/verbal_report.txt")
        print(f"HTML отчет сохранен в: {output_dir}/summary_report.html")

        return summary, total_score

    return None, 0

# ========================================================
# ОСНОВНОЙ БЛОК
# ========================================================

if __name__ == "__main__":
    # ========================================================
    # НАСТРОЙКИ
    # ========================================================

    SYNTHETIC_IMAGES_DIR = "/data/training/train"
    OUTPUT_DIR = "expert_reports"
    BATCH_SAMPLE_SIZE = 50

    # ========================================================
    # ЗАПУСК
    # ========================================================

    print("=" * 70)
    print("ОЦЕНКА КАЧЕСТВА СИНТЕТИЧЕСКОЙ ВЫБОРКИ С ОТЧЕТОМ")
    print("=" * 70)

    summary, total_score = batch_evaluate_with_report(
        SYNTHETIC_IMAGES_DIR, OUTPUT_DIR, BATCH_SAMPLE_SIZE
    )

    if summary:
        print(f"\nОБЩИЙ БАЛЛ ВЫБОРКИ: {total_score:.1f}/10")

        # Интерпретация балла
        if total_score >= 9.0:
            print("✅ ВЫСОЧАЙШЕЕ КАЧЕСТВО - выборка превосходна для обучения")
        elif total_score >= 7.0:
            print("✅ ХОРОШЕЕ КАЧЕСТВО - выборка пригодна для производственного использования")
        elif total_score >= 5.0:
            print("⚠️ СРЕДНЕЕ КАЧЕСТВО - выборка требует улучшений")
        elif total_score >= 3.0:
            print("❌ НИЗКОЕ КАЧЕСТВО - требуется значительная доработка")
        else:
            print("🚨 КРИТИЧЕСКО НИЗКОЕ КАЧЕСТВО - выборка непригодна")
    else:
        print("Не удалось выполнить оценку выборки")