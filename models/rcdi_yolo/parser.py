"""
models/rcdi_yolo/parser.py

Parses RCDI-YOLO configuration YAML into constructor arguments.
"""
import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    logger.info("Loaded config: %s", path)
    return cfg


def parse_rcdi_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse rcdi_yolo_1c.yaml into kwargs for RCDIYOLOModel.

    Args:
        cfg: Parsed YAML dict from configs/rcdi_yolo_1c.yaml.

    Returns:
        Dict of constructor arguments.
    """
    model_cfg    = cfg.get("model", {})
    lan_cfg      = cfg.get("lanconvnext", {})
    dysample_cfg = cfg.get("dysample", {})
    head_cfg     = cfg.get("implicit_head", {})

    return {
        "in_channels":            int(model_cfg.get("input_channels", 1)),
        "num_classes":            int(model_cfg.get("num_classes", 1)),
        "base_channels":          list(model_cfg.get("base_channels", [64, 128, 256, 512, 512])),
        "width_multiple":         float(model_cfg.get("width_multiple", 0.5)),
        "depth_multiple":         float(model_cfg.get("depth_multiple", 0.33)),
        "reg_max":                int(model_cfg.get("reg_max", 16)),
        "backbone_lan_positions": list(lan_cfg.get("backbone_positions", [1, 2])),
        "neck_lan_positions":     list(lan_cfg.get("neck_positions", [1, 2, 4])),
        "dysample_cfg": {
            "enabled": bool(dysample_cfg.get("enabled", True)),
            "scale":   int(dysample_cfg.get("scale", 2)),
            "style":   str(dysample_cfg.get("style", "lp")),
            "groups":  int(dysample_cfg.get("groups", 4)),
        },
        "use_implicit_head": bool(head_cfg.get("enabled", True)),
    }


def get_model_kwargs_from_file(config_path: str) -> Dict[str, Any]:
    """Convenience: load config file and parse in one call."""
    return parse_rcdi_config(load_config(config_path))
