"""FSOT 2.1 LLM package — language models as local observers in fluid spacetime."""

__version__ = "0.1.0"
__fsot__ = "Fluid Spacetime Omni-Theory is the intrinsic model of everything."

from .paths import workspace_root, ensure_sys_path

ensure_sys_path()

__all__ = ["__version__", "__fsot__", "workspace_root", "ensure_sys_path"]
