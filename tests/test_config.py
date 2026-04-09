from __future__ import annotations

import os
from pathlib import Path

import pytest

from rottengenizdat.config import (
    CONFIG_DIR,
    load_config,
    save_config,
    resolve_slack_token,
    resolve_slack_channel,
    config_set,
)


class TestLoadConfig:
    def test_returns_empty_dict_when_no_file(self, tmp_path: Path):
        config = load_config(config_dir=tmp_path)
        assert config == {}

    def test_reads_existing_config(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[slack]\ntoken = "xoxb-test"\nchannel = "C123"\n')
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "xoxb-test"
        assert config["slack"]["channel"] == "C123"


class TestSaveConfig:
    def test_creates_file_and_dirs(self, tmp_path: Path):
        nested = tmp_path / "sub" / "dir"
        save_config({"slack": {"token": "xoxb-abc"}}, config_dir=nested)
        assert (nested / "config.toml").exists()
        config = load_config(config_dir=nested)
        assert config["slack"]["token"] == "xoxb-abc"

    def test_overwrites_existing(self, tmp_path: Path):
        save_config({"slack": {"token": "old"}}, config_dir=tmp_path)
        save_config({"slack": {"token": "new"}}, config_dir=tmp_path)
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "new"


class TestConfigSet:
    def test_set_nested_key(self, tmp_path: Path):
        config_set("slack.token", "xoxb-set-test", config_dir=tmp_path)
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "xoxb-set-test"

    def test_set_preserves_existing_keys(self, tmp_path: Path):
        config_set("slack.token", "xoxb-keep", config_dir=tmp_path)
        config_set("slack.channel", "C999", config_dir=tmp_path)
        config = load_config(config_dir=tmp_path)
        assert config["slack"]["token"] == "xoxb-keep"
        assert config["slack"]["channel"] == "C999"


class TestResolveSlackToken:
    def test_prefers_config_file(self, tmp_path: Path, monkeypatch):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[slack]\ntoken = "xoxb-from-file"\n')
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
        token = resolve_slack_token(config_dir=tmp_path)
        assert token == "xoxb-from-file"

    def test_falls_back_to_env(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-from-env")
        token = resolve_slack_token(config_dir=tmp_path)
        assert token == "xoxb-from-env"

    def test_raises_when_no_token(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        with pytest.raises(ValueError, match="Slack token not found"):
            resolve_slack_token(config_dir=tmp_path)


class TestResolveSlackChannel:
    def test_from_config(self, tmp_path: Path):
        config_file = tmp_path / "config.toml"
        config_file.write_text('[slack]\nchannel = "C456"\n')
        channel = resolve_slack_channel(config_dir=tmp_path)
        assert channel == "C456"

    def test_raises_when_no_channel(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Slack channel not configured"):
            resolve_slack_channel(config_dir=tmp_path)
