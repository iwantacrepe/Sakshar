"""Lazy model registry with split GPU residency.

On a 6 GB RTX 3060 all voice models fit at once in reduced precision:
    faster-whisper small (int8_float16, CTranslate2)  ~1.0 GB
    IndicTrans2 dist-200M en->indic + indic->en fp16  ~1.2 GB
    Indic Parler-TTS fp16                             ~2.5 GB
So models stay resident after first use (no per-request PCIe shuffling). Eviction happens
only when CUDA actually runs out of memory: other torch models move to CPU and the load retries,
and as a last resort the requested model loads on CPU.
"""
from __future__ import annotations

import gc
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import get_settings

log = logging.getLogger(__name__)


@dataclass
class Loaded:
    obj: Any
    device: str
    movable: bool  # torch modules can be moved to CPU; CTranslate2 models cannot
    lock: threading.Lock = field(default_factory=threading.Lock)  # one inference at a time per model


class Registry:
    def __init__(self) -> None:
        self._loaders: dict[str, tuple[Callable[[str], Any], bool]] = {}
        self._models: dict[str, Loaded] = {}
        self._lock = threading.RLock()

    def register(self, name: str, loader: Callable[[str], Any], movable: bool = True) -> None:
        self._loaders[name] = (loader, movable)

    def device(self) -> str:
        if get_settings().device != "cuda":
            return "cpu"
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def get(self, name: str) -> Loaded:
        with self._lock:
            if name in self._models:
                return self._models[name]
            loader, movable = self._loaders[name]
            dev = self.device()
            try:
                obj = loader(dev)
            except Exception as e:
                if dev == "cuda" and _is_oom(e):
                    log.warning("CUDA OOM loading %s; offloading other models", name)
                    self._offload_others(name)
                    try:
                        obj = loader(dev)
                    except Exception as e2:
                        if not _is_oom(e2):
                            raise
                        log.warning("Still OOM; loading %s on CPU", name)
                        dev = "cpu"
                        obj = loader(dev)
                else:
                    raise
            self._models[name] = Loaded(obj=obj, device=dev, movable=movable)
            log.info("loaded %s on %s", name, dev)
            return self._models[name]

    def _offload_others(self, keep: str) -> None:
        for n, m in list(self._models.items()):
            if n == keep or m.device != "cuda":
                continue
            if m.movable and hasattr(m.obj, "to"):
                m.obj.to("cpu")
                m.device = "cpu"
            elif m.movable and isinstance(m.obj, tuple):
                for part in m.obj:
                    if hasattr(part, "to") and hasattr(part, "parameters"):
                        part.to("cpu")
                m.device = "cpu"
            else:
                del self._models[n]
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except ImportError:
            pass

    def status(self) -> dict[str, str]:
        return {n: m.device for n, m in self._models.items()}


def _is_oom(e: Exception) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or type(e).__name__ == "OutOfMemoryError"


registry = Registry()
