# src/utils/checkpoint_manager.py
import os
import json
import torch
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Универсальный менеджер чекпоинтов для ML-обучения"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5, verbose: bool = True):
        """
        Args:
            checkpoint_dir: Директория для чекпоинтов
            max_checkpoints: Максимальное количество хранимых чекпоинтов
            verbose: Подробный вывод
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.verbose = verbose
        self._checkpoints_metadata = []
        
        # Загружаем метаданные если есть
        self.metadata_path = self.checkpoint_dir / "checkpoints_meta.json"
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self._checkpoints_metadata = json.load(f)
    
    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        fold: int = 0,
        metrics: Optional[Dict] = None,
        is_best: bool = False,
        is_last: bool = False
    ) -> str:
        """
        Сохраняет чекпоинт обучения
        
        Args:
            state_dict: Словарь с состоянием (модель, оптимизатор, scheduler)
            epoch: Номер эпохи
            fold: Номер фолда (для кросс-валидации)
            metrics: Метрики на момент сохранения
            is_best: Является ли эта модель лучшей
            is_last: Это последняя эпоха
        
        Returns:
            Путь к сохранённому файлу
        """
        try:
            # Создаём имя файла
            if is_best:
                filename = f"best_fold{fold}.pth"
            elif is_last:
                filename = f"last_fold{fold}.pth"
            else:
                filename = f"checkpoint_fold{fold}_epoch{epoch:04d}.pth"
            
            checkpoint_path = self.checkpoint_dir / filename
            
            # Добавляем метаданные
            checkpoint_data = {
                'state_dict': state_dict,
                'epoch': epoch,
                'fold': fold,
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics or {},
                'is_best': is_best,
                'is_last': is_last
            }
            
            # Сохраняем
            torch.save(checkpoint_data, checkpoint_path)
            
            # Обновляем метаданные
            checkpoint_meta = {
                'filename': filename,
                'epoch': epoch,
                'fold': fold,
                'timestamp': checkpoint_data['timestamp'],
                'metrics': metrics,
                'is_best': is_best,
                'is_last': is_last,
                'file_size': checkpoint_path.stat().st_size
            }
            
            self._update_metadata(checkpoint_meta)
            
            if self.verbose:
                logger.info(f"💾 Чекпоинт сохранён: {checkpoint_path}")
                if metrics:
                    logger.info(f"   Метрики: {metrics}")
            
            return str(checkpoint_path)
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения чекпоинта: {e}")
            raise
    
    def _update_metadata(self, new_meta: Dict):
        """Обновляет метаданные чекпоинтов"""
        # Удаляем старые чекпоинты если превышен лимит
        if len(self._checkpoints_metadata) >= self.max_checkpoints:
            # Ищем не-best и не-last чекпоинты для удаления
            regular_checkpoints = [
                (i, meta) for i, meta in enumerate(self._checkpoints_metadata)
                if not meta.get('is_best', False) and not meta.get('is_last', False)
            ]
            
            if regular_checkpoints:
                # Сортируем по эпохе (старые первыми)
                regular_checkpoints.sort(key=lambda x: x[1]['epoch'])
                idx_to_remove, meta_to_remove = regular_checkpoints[0]
                
                # Удаляем файл
                file_to_remove = self.checkpoint_dir / meta_to_remove['filename']
                if file_to_remove.exists():
                    file_to_remove.unlink()
                
                # Удаляем из метаданных
                self._checkpoints_metadata.pop(idx_to_remove)
                logger.debug(f"Удалён старый чекпоинт: {meta_to_remove['filename']}")
        
        # Добавляем новые метаданные
        self._checkpoints_metadata.append(new_meta)
        
        # Сохраняем метаданные
        with open(self.metadata_path, 'w') as f:
            json.dump(self._checkpoints_metadata, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, 
                       fold: int = 0, epoch: Optional[int] = None,
                       best: bool = False, last: bool = False) -> Optional[Dict]:
        """
        Загружает чекпоинт
        
        Args:
            checkpoint_path: Прямой путь к файлу (если None, ищем по fold/epoch/best/last)
            fold: Номер фолда для поиска
            epoch: Конкретная эпоха для загрузки
            best: Загрузить лучшую модель для фолда
            last: Загрузить последнюю модель для фолда
        
        Returns:
            Загруженный чекпоинт или None
        """
        try:
            # Определяем путь к файлу
            if checkpoint_path:
                filepath = Path(checkpoint_path)
            elif best:
                filepath = self.checkpoint_dir / f"best_fold{fold}.pth"
            elif last:
                filepath = self.checkpoint_dir / f"last_fold{fold}.pth"
            elif epoch is not None:
                filepath = self.checkpoint_dir / f"checkpoint_fold{fold}_epoch{epoch:04d}.pth"
            else:
                # Ищем последний чекпоинт для фолда
                filepath, _ = self.find_latest_checkpoint(fold)
                if not filepath:
                    return None
            
            if not filepath.exists():
                logger.warning(f"Файл чекпоинта не найден: {filepath}")
                return None
            
            # Загружаем
            checkpoint = torch.load(filepath, map_location='cpu')
            
            if self.verbose:
                logger.info(f"📥 Загружен чекпоинт: {filepath.name}")
                logger.info(f"   Эпоха: {checkpoint.get('epoch', 'N/A')}")
                if checkpoint.get('metrics'):
                    logger.info(f"   Метрики: {checkpoint['metrics']}")
            
            return checkpoint
            
        except Exception as e:
            logger.error(f"Ошибка загрузки чекпоинта: {e}")
            return None
    
    def find_latest_checkpoint(self, fold: int = 0) -> Tuple[Optional[Path], int]:
        """
        Находит последний чекпоинт для фолда
        
        Returns:
            (путь к файлу, номер эпохи) или (None, -1)
        """
        try:
            # Ищем в метаданных
            fold_checkpoints = [
                meta for meta in self._checkpoints_metadata
                if meta.get('fold') == fold and not meta.get('is_best', False)
            ]
            
            if not fold_checkpoints:
                # Ищем по файлам
                pattern = f"checkpoint_fold{fold}_epoch*.pth"
                checkpoint_files = list(self.checkpoint_dir.glob(pattern))
                
                if not checkpoint_files:
                    return None, -1
                
                # Извлекаем номер эпохи из имени файла
                latest_file = None
                latest_epoch = -1
                
                for file in checkpoint_files:
                    try:
                        # Имя: checkpoint_fold0_epoch0100.pth
                        name = file.stem
                        epoch_str = name.split('_epoch')[1]
                        epoch = int(epoch_str)
                        
                        if epoch > latest_epoch:
                            latest_epoch = epoch
                            latest_file = file
                    except (ValueError, IndexError):
                        continue
                
                return latest_file, latest_epoch
            
            # Используем метаданные
            fold_checkpoints.sort(key=lambda x: x['epoch'], reverse=True)
            latest_meta = fold_checkpoints[0]
            filepath = self.checkpoint_dir / latest_meta['filename']
            
            return filepath, latest_meta['epoch']
            
        except Exception as e:
            logger.error(f"Ошибка поиска чекпоинтов: {e}")
            return None, -1
    
    def get_checkpoint_info(self) -> Dict:
        """Возвращает информацию о всех чекпоинтах"""
        return {
            'checkpoint_dir': str(self.checkpoint_dir),
            'total_checkpoints': len(self._checkpoints_metadata),
            'checkpoints': self._checkpoints_metadata,
            'best_models': [
                meta for meta in self._checkpoints_metadata 
                if meta.get('is_best', False)
            ],
            'last_models': [
                meta for meta in self._checkpoints_metadata 
                if meta.get('is_last', False)
            ]
        }
    
    def cleanup(self, keep_best: bool = True, keep_last: bool = True):
        """Очищает чекпоинты, оставляя только best/last если нужно"""
        try:
            if not self.checkpoint_dir.exists():
                return
            
            # Собираем файлы для удаления
            files_to_keep = set()
            
            if keep_best:
                best_files = [meta['filename'] for meta in self._checkpoints_metadata 
                            if meta.get('is_best', False)]
                files_to_keep.update(best_files)
            
            if keep_last:
                last_files = [meta['filename'] for meta in self._checkpoints_metadata 
                            if meta.get('is_last', False)]
                files_to_keep.update(last_files)
            
            # Удаляем остальные
            for file in self.checkpoint_dir.glob("*.pth"):
                if file.name not in files_to_keep:
                    file.unlink()
                    logger.debug(f"Удалён чекпоинт: {file.name}")
            
            # Обновляем метаданные
            self._checkpoints_metadata = [
                meta for meta in self._checkpoints_metadata
                if meta['filename'] in files_to_keep
            ]
            
            with open(self.metadata_path, 'w') as f:
                json.dump(self._checkpoints_metadata, f, indent=2)
            
            logger.info("🗑️ Очистка чекпоинтов завершена")
            
        except Exception as e:
            logger.error(f"Ошибка очистки чекпоинтов: {e}")