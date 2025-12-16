# src/generators/__init__.py
from .crack import make_a_crack
from .pore import make_pore
from .incomplete_fusion import make_incomplete_fusion
from .empty import make_empty_seam

# Упрощённый интерфейс
def generate_defect(defect_type: str, seed: int = None):
    """Универсальный генератор дефектов"""
    generators = {
        'crack': make_a_crack,
        'pore': make_pore,
        'fusion': make_incomplete_fusion,
        'empty': make_empty_seam
    }

    if defect_type not in generators:
        raise ValueError(f"Неизвестный тип дефекта: {defect_type}")

    gen = generators[defect_type]()
    return gen.generate(seed=seed)