from __future__ import annotations

import os
import tomllib
from pathlib import Path

import tomli_w

CONFIG_DIR = Path.home() / ".config" / "rottengenizdat"
CONFIG_FILENAME = "config.toml"


def _config_path(config_dir: Path = CONFIG_DIR) -> Path:
    return config_dir / CONFIG_FILENAME


def load_config(config_dir: Path = CONFIG_DIR) -> dict:
    """Load config from TOML file. Returns empty dict if file doesn't exist."""
    path = _config_path(config_dir)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def save_config(config: dict, config_dir: Path = CONFIG_DIR) -> None:
    """Write config dict to TOML file, creating directories as needed."""
    path = _config_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        tomli_w.dump(config, f)


def config_set(key: str, value: str, config_dir: Path = CONFIG_DIR) -> None:
    """Set a dotted key (e.g. 'slack.token') in the config file."""
    config = load_config(config_dir)
    parts = key.split(".")
    d = config
    for part in parts[:-1]:
        d = d.setdefault(part, {})
    d[parts[-1]] = value
    save_config(config, config_dir)


def resolve_slack_token(config_dir: Path = CONFIG_DIR) -> str:
    """Resolve Slack token: config file > SLACK_BOT_TOKEN env var > error."""
    config = load_config(config_dir)
    token = config.get("slack", {}).get("token")
    if token:
        return token
    token = os.environ.get("SLACK_BOT_TOKEN")
    if token:
        return token
    raise ValueError(
        "Slack token not found. Set it with:\n"
        "  rotten config set slack.token xoxb-YOUR-TOKEN\n"
        "or set the SLACK_BOT_TOKEN environment variable."
    )


def resolve_slack_channel(config_dir: Path = CONFIG_DIR) -> str:
    """Resolve Slack channel ID from config file."""
    config = load_config(config_dir)
    channel = config.get("slack", {}).get("channel")
    if channel:
        return channel
    raise ValueError(
        "Slack channel not configured. Set it with:\n"
        "  rotten config set slack.channel C0XXXXXXX"
    )
