from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rottengenizdat.sample_sale import (
    CACHE_DIR,
    IndexEntry,
    load_index,
    save_index,
    extract_media_from_messages,
    pick_random_samples,
)


@pytest.fixture
def sample_index() -> list[dict]:
    return [
        {
            "id": "F001",
            "type": "attachment",
            "filename": "beat.wav",
            "mime": "audio/wav",
            "slack_url": "https://files.slack.com/beat.wav",
            "cached_path": "samples/F001-beat.wav",
            "message_ts": "1700000001.000100",
        },
        {
            "id": "sha256-abc123",
            "type": "link",
            "url": "https://youtube.com/watch?v=abc",
            "cached_path": "samples/sha256-abc123.wav",
            "message_ts": "1700000002.000200",
        },
    ]


class TestIndexIO:
    def test_load_empty(self, tmp_path: Path):
        entries = load_index(cache_dir=tmp_path)
        assert entries == []

    def test_save_and_load(self, tmp_path: Path, sample_index):
        entries = [IndexEntry(**e) for e in sample_index]
        save_index(entries, cache_dir=tmp_path)
        loaded = load_index(cache_dir=tmp_path)
        assert len(loaded) == 2
        assert loaded[0].id == "F001"
        assert loaded[1].type == "link"


class TestExtractMedia:
    def test_extracts_audio_attachment(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "files": [
                    {
                        "id": "F001",
                        "name": "beat.wav",
                        "mimetype": "audio/wav",
                        "url_private_download": "https://files.slack.com/beat.wav",
                    }
                ],
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 1
        assert entries[0].type == "attachment"
        assert entries[0].filename == "beat.wav"

    def test_skips_non_audio_attachment(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "files": [
                    {
                        "id": "F002",
                        "name": "photo.png",
                        "mimetype": "image/png",
                        "url_private_download": "https://files.slack.com/photo.png",
                    }
                ],
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 0

    def test_extracts_video_attachment(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "files": [
                    {
                        "id": "F003",
                        "name": "clip.mp4",
                        "mimetype": "video/mp4",
                        "url_private_download": "https://files.slack.com/clip.mp4",
                    }
                ],
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 1

    def test_extracts_url_links(self):
        messages = [
            {
                "ts": "1700000003.000300",
                "text": "check this out https://youtube.com/watch?v=xyz",
            }
        ]
        entries = extract_media_from_messages(messages)
        assert len(entries) == 1
        assert entries[0].type == "link"
        assert "youtube.com" in entries[0].url

    def test_no_duplicate_ids(self):
        messages = [
            {
                "ts": "1700000001.000100",
                "text": "https://youtube.com/watch?v=xyz",
            },
            {
                "ts": "1700000002.000200",
                "text": "https://youtube.com/watch?v=xyz",
            },
        ]
        entries = extract_media_from_messages(messages)
        ids = [e.id for e in entries]
        assert len(set(ids)) == len(ids)


class TestPickRandom:
    def test_picks_correct_count(self, sample_index):
        entries = [IndexEntry(**e) for e in sample_index]
        picked = pick_random_samples(entries, count=1)
        assert len(picked) == 1

    def test_picks_all_when_count_exceeds(self, sample_index):
        entries = [IndexEntry(**e) for e in sample_index]
        picked = pick_random_samples(entries, count=10)
        assert len(picked) == 2

    def test_empty_index_raises(self):
        with pytest.raises(ValueError, match="No samples"):
            pick_random_samples([], count=1)


from typer.testing import CliRunner
from rottengenizdat.cli import app

runner = CliRunner()


class TestSampleSaleCLI:
    def test_list_empty(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("rottengenizdat.sample_sale.CACHE_DIR", tmp_path)
        result = runner.invoke(app, ["sample-sale", "list"])
        assert result.exit_code == 0
        assert "No samples" in result.stdout or "no samples" in result.stdout.lower() or "Run" in result.stdout

    def test_list_with_entries(self, tmp_path: Path, monkeypatch, sample_index):
        monkeypatch.setattr("rottengenizdat.sample_sale.CACHE_DIR", tmp_path)
        entries = [IndexEntry(**e) for e in sample_index]
        save_index(entries, cache_dir=tmp_path)
        result = runner.invoke(app, ["sample-sale", "list"])
        assert result.exit_code == 0
        assert "beat.wav" in result.stdout

    def test_clear(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr("rottengenizdat.sample_sale.CACHE_DIR", tmp_path)
        samples_dir = tmp_path / "samples"
        samples_dir.mkdir()
        (samples_dir / "fake.wav").write_bytes(b"fake")
        result = runner.invoke(app, ["sample-sale", "clear"])
        assert result.exit_code == 0
        assert not samples_dir.exists()
