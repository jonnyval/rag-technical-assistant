import time
from src.logger import log
from src.config import settings

class TimeProfiler:
    """
    Контекстный менеджер для замера времени выполнения участков кода.
    Включается и выключается через settings.enable_profiling.
    """
    def __init__(self, step_name: str):
        """Запоминает название профилируемого шага."""

        self.step_name = step_name
        self.start_time = 0.0

    def __enter__(self):
        """Фиксирует время начала шага, если профилирование включено."""

        if settings.enable_profiling:
            self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Логирует длительность шага при выходе из контекстного менеджера."""

        if settings.enable_profiling:
            elapsed = time.perf_counter() - self.start_time
            # Выводим время в миллисекундах для большей наглядности
            log.info(f"⏱ [ПРОФАЙЛЕР] {self.step_name} | Заняло: {elapsed:.4f} сек. ({elapsed * 1000:.0f} мс)")
