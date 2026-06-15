"""Run one of the JSON experiment presets from the configs directory."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_path", help="Path to a JSON config file.")
    return parser.parse_args()


def append_flag(command: list[str], flag_name: str, value: object) -> None:
    """Translate a JSON config value into CLI flags."""
    cli_flag = f"--{flag_name}"

    if value is None or value is False:
        return

    if value is True:
        command.append(cli_flag)
        return

    if isinstance(value, list):
        for item in value:
            command.extend([cli_flag, str(item)])
        return

    command.extend([cli_flag, str(value)])


def load_config(config_path: Path) -> dict[str, object]:
    """Read one preset config from disk."""
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    config_path = (PROJECT_ROOT / args.config_path).resolve()
    config = load_config(config_path)

    script_name = str(config["script"])
    script_path = (PROJECT_ROOT / "scripts" / script_name).resolve()
    cli_args = config.get("args", {})
    if not isinstance(cli_args, dict):
        raise ValueError("Config field 'args' must be a JSON object.")

    command = [sys.executable, str(script_path)]
    for key, value in cli_args.items():
        append_flag(command, key, value)

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

