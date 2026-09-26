"""
main.py — NEMO entry point.

Usage:
    python main.py --mode recorded --config configs/config.yaml --mission path/to/mission/
    python main.py --mode api --config configs/config.yaml
    streamlit run dashboard/app.py  (legacy, for frontend team reference)
"""
import argparse
import logging
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nemo")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_recorded(config: dict, mission_dir: str) -> None:
    """Run inference pipeline on a recorded mission directory."""
    from mission.recorded import RecordedMissionSource
    from inference.pipeline import MissionPipeline
    from reporting.report_generator import ReportGenerator

    logger.info("Mode: recorded | Mission: %s", mission_dir)

    source = RecordedMissionSource(mission_dir)

    if len(source) == 0:
        logger.warning("No sonar frames found in %s", mission_dir)
        return

    pipeline = MissionPipeline.from_config(
        config,
        navigation_file=source.navigation_file,
    )

    def progress(idx, total):
        if idx % 10 == 0 or idx == total:
            logger.info("Progress: %d/%d frames", idx, total)

    all_results = pipeline.process_mission(source, progress_callback=progress)

    reporter = ReportGenerator(config.get("output", {}))
    outputs = reporter.generate(all_results, mission_id=source.mission_id)

    logger.info("Reports written:")
    for fmt, path in outputs.items():
        logger.info("  %s: %s", fmt, path)


def run_api(config: dict) -> None:
    """Start the NEMO REST API server."""
    from dashboard.app import app
    import uvicorn

    host = config.get("dashboard", {}).get("host", "0.0.0.0")
    port = config.get("dashboard", {}).get("port", 8000)
    logger.info("Starting NEMO REST API at http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NEMO — AI-powered side-scan sonar debris detection"
    )
    parser.add_argument("--mode", choices=["recorded", "api"], default="recorded")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mission", default=None, help="Path to mission directory")
    args = parser.parse_args()

    if not Path(args.config).exists():
        logger.error("Config not found: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)

    if args.mode == "recorded":
        if not args.mission:
            logger.error("--mission path required for recorded mode")
            sys.exit(1)
        run_recorded(config, args.mission)
    elif args.mode == "api":
        run_api(config)


if __name__ == "__main__":
    main()
