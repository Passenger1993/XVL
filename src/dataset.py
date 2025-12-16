import random
import numpy as np
from PIL import Image
from .generators import *

# Оптимизированные версии генераторов
def make_incomplete_fusion_optimized(seed=None):
    """Оптимизированная версия генератора неполного провара"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    try:
        return make_incomplete_fusion()
    except Exception:
        img = Image.new('L', (CFG.IMG_SIZE, CFG.IMG_SIZE), color=128)
        bbox_dict = {}
        return img, bbox_dict

def make_a_crack_optimized(seed=None):
    """Оптимизированная версия генератора трещин"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    try:
        return make_a_crack()
    except Exception:
        img = Image.new('L', (CFG.IMG_SIZE, CFG.IMG_SIZE), color=128)
        bbox_dict = {}
        return img, bbox_dict

def make_pore_optimized(num_pores=1, seed=None):
    """Оптимизированная версия генератора пор"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    try:
        return make_pore(num_pores=num_pores)
    except Exception:
        img = Image.new('L', (CFG.IMG_SIZE, CFG.IMG_SIZE), color=128)
        bbox_dict = {}
        return img, bbox_dict

def make_empty_seam_optimized(seed=None):
    """Оптимизированная версия генератора пустых швов"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    try:
        return make_empty_seam()
    except Exception:
        return Image.new('L', (CFG.IMG_SIZE, CFG.IMG_SIZE), color=128)

class ParallelSampleGenerator:
    """Параллельный генератор образцов с оптимизированной обработкой"""

    def __init__(self, num_samples, img_size, mode='train', num_workers=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.mode = mode
        self.num_workers = num_workers or min(CFG.NUM_WORKERS, os.cpu_count() - 1 or 1)

        # Используем numpy для генерации всех class_id сразу
        np.random.seed(42 if mode == 'val' else None)
        self.class_ids = np.random.choice(5, size=num_samples, p=CFG.CLASS_WEIGHTS)

        # Группируем по классам для пакетной обработки
        self.class_indices = {}
        for i, class_id in enumerate(self.class_ids):
            self.class_indices.setdefault(class_id, []).append(i)

    def _generate_class_batch(self, class_id, indices, process_idx):
        """Генерация батча изображений одного класса"""
        results = []
        base_seed = 42 if self.mode == 'val' else int(time.time() * 1000) % 1000000

        for i, idx in enumerate(indices):
            seed = base_seed + idx if self.mode == 'val' else None

            try:
                if class_id == 0:
                    img, bbox_dict = make_incomplete_fusion_optimized(seed)
                elif class_id == 1:
                    img, bbox_dict = make_a_crack_optimized(seed)
                elif class_id == 2:
                    img, bbox_dict = make_pore_optimized(num_pores=1, seed=seed)
                elif class_id == 3:
                    num_pores = np.random.randint(3, 8)
                    img, bbox_dict = make_pore_optimized(num_pores=num_pores, seed=seed)
                else:
                    img = make_empty_seam_optimized(seed)
                    bbox_dict = {}

                # Оптимизированная обработка изображения
                if isinstance(img, Image.Image):
                    if img.mode == 'L':
                        img_np = np.array(img)
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                    else:
                        img_np = np.array(img.convert('RGB'))

                    img_np = cv2.resize(img_np, (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_AREA)
                else:
                    img_np = cv2.resize(img, (self.img_size, self.img_size),
                                      interpolation=cv2.INTER_AREA)
                    if len(img_np.shape) == 2:
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

                # Подготовка bounding boxes
                bboxes = []
                if bbox_dict:
                    for bbox in bbox_dict.values():
                        if len(bbox) == 4:
                            x_min, y_min, x_max, y_max = bbox
                            if x_min < x_max and y_min < y_max:
                                x_center = ((x_min + x_max) / 2) / self.img_size
                                y_center = ((y_min + y_max) / 2) / self.img_size
                                width = (x_max - x_min) / self.img_size
                                height = (y_max - y_min) / self.img_size

                                if (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                                    0 < width <= 1 and 0 < height <= 1):
                                    bboxes.append([class_id, x_center, y_center, width, height])

                results.append((idx, img_np, bboxes))

            except Exception:
                img_np = np.full((self.img_size, self.img_size, 3), 128, dtype=np.uint8)
                results.append((idx, img_np, []))

        return results

    def generate_all_samples(self):
        """Генерация всех образцов с параллелизацией"""
        from concurrent.futures import ProcessPoolExecutor, as_completed

        all_images = np.zeros((self.num_samples, self.img_size, self.img_size, 3), dtype=np.uint8)
        all_labels = [None] * self.num_samples

        print(f"Генерация {self.num_samples} изображений...")
        start_time = time.time()

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []

            for class_id, indices in self.class_indices.items():
                for i in range(0, len(indices), CFG.GENERATION_BATCH_SIZE):
                    batch_indices = indices[i:i + CFG.GENERATION_BATCH_SIZE]
                    futures.append(executor.submit(
                        self._generate_class_batch,
                        class_id, batch_indices, len(futures)
                    ))

            completed = 0
            for future in futures:
                batch_results = future.result()
                for idx, img_np, bboxes in batch_results:
                    all_images[idx] = img_np
                    all_labels[idx] = bboxes if bboxes else []
                    completed += 1

        elapsed = time.time() - start_time
        print(f"Готово: {completed} изображений за {elapsed:.1f} сек ({completed/elapsed:.1f} img/сек)")

        return all_images, all_labels

class DynamicDefectDataset(Dataset):
    """Датасет с динамической генерацией дефектов на лету"""

    def __init__(self, num_samples, img_size=CFG.IMG_SIZE, mode='train',
                 pre_generated_data=None):
        self.num_samples = num_samples
        self.img_size = img_size
        self.mode = mode
        self.classes = {
            0: 'incomplete_fusion',
            1: 'crack',
            2: 'single_pore',
            3: 'cluster_pores',
            4: 'empty'
        }
        self.class_weights = CFG.CLASS_WEIGHTS

        if pre_generated_data is not None:
            self.images, self.labels = pre_generated_data
            self.pre_generated = True
        else:
            self.pre_generated = False
            self.images = None
            self.labels = None

            if mode == 'val':
                random.seed(42)
                np.random.seed(42)
                torch.manual_seed(42)

    def __len__(self):
        return self.num_samples

    def _generate_if_needed(self):
        """Генерация данных при первом обращении"""
        if self.images is None:
            generator = ParallelSampleGenerator(
                self.num_samples, self.img_size, self.mode
            )
            self.images, self.labels = generator.generate_all_samples()

    def __getitem__(self, idx):
        if not self.pre_generated:
            self._generate_if_needed()

        if self.mode == 'val' and not self.pre_generated:
            random.seed(42 + idx)
            np.random.seed(42 + idx)

        img_np = self.images[idx]
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        if self.labels[idx]:
            label_tensor = torch.FloatTensor(self.labels[idx])
        else:
            label_tensor = torch.zeros((0, 5))

        return img_tensor, label_tensor

    def save_to_disk(self, images_dir, labels_dir, prefix='train'):
        """Эффективное сохранение всех изображений на диск"""
        import threading
        from queue import Queue

        if self.images is None:
            self._generate_if_needed()

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        print(f"Сохранение {len(self.images)} изображений...")
        start_time = time.time()

        write_queue = Queue()

        def writer_thread():
            """Поток для записи файлов"""
            while True:
                task = write_queue.get()
                if task is None:
                    break

                idx, img_np, labels = task
                img_path = os.path.join(images_dir, f'{prefix}_{idx:06d}.jpg')
                cv2.imwrite(img_path, img_np, [cv2.IMWRITE_JPEG_QUALITY, 95])

                if labels:
                    label_path = os.path.join(labels_dir, f'{prefix}_{idx:06d}.txt')
                    with open(label_path, 'w') as f:
                        for label in labels:
                            class_id, x_center, y_center, width, height = label
                            f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                write_queue.task_done()

        num_writer_threads = min(4, os.cpu_count() // 2 or 1)
        writers = []
        for _ in range(num_writer_threads):
            t = threading.Thread(target=writer_thread)
            t.start()
            writers.append(t)

        for idx in range(len(self.images)):
            write_queue.put((idx, self.images[idx], self.labels[idx]))

        write_queue.join()

        for _ in range(num_writer_threads):
            write_queue.put(None)

        for t in writers:
            t.join()

        elapsed = time.time() - start_time
        print(f"Изображения сохранены за {elapsed:.1f} сек")
