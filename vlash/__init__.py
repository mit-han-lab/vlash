"""
vlash: Real-Time VLAs via Future-state-aware Asynchronous Inference
"""

__version__ = "0.1.0"

__all__ = ["configs", "datasets", "__version__"]


def __getattr__(name: str):
    if name in ("configs", "datasets"):
        import importlib

        mod = importlib.import_module(f"vlash.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module 'vlash' has no attribute {name!r}")

