"""
models/rcdi_yolo/head.py

Re-exports ImplicitHead for clean imports from the model package.
See models/rcdi_yolo/modules/implicit_head.py for the full implementation.
"""
from models.rcdi_yolo.modules.implicit_head import ImplicitHead, HeadBranch, DFL

__all__ = ["ImplicitHead", "HeadBranch", "DFL"]
