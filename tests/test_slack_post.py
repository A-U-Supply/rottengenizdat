from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rottengenizdat.slack_post import (
    _ts_to_date,
    format_main_comment,
    format_thread_reply,
    resolve_display_name,
)


class TestFormatMainComment:
    def test_with_label_only(self):
        result = format_main_comment("bone-xray")
        assert result == ":radio: *rottengenizdat: bone-xray*"

    def test_with_label_and_run_url(self):
        result = format_main_comment("fever-dream", "https://github.com/runs/123")
        assert ":radio: *rottengenizdat: fever-dream*" in result
        assert "<https://github.com/runs/123|View logs>" in result


class TestFormatThreadReply:
    def test_single_source(self):
        sources = [
            {
                "display_name": "jake",
                "date": "2026-03-15",
                "permalink": "https://slack.com/archives/C123/p456",
            }
        ]
        result = format_thread_reply(sources)
        assert "source:" in result
        assert "jake (2026-03-15)" in result
        assert "#sample-sale" in result
        assert "original:" in result
        assert "<https://slack.com/archives/C123/p456|view>" in result

    def test_multiple_sources(self):
        sources = [
            {
                "display_name": "jake",
                "date": "2026-03-15",
                "permalink": "https://slack.com/p1",
            },
            {
                "display_name": "alice",
                "date": "2026-03-16",
                "permalink": "https://slack.com/p2",
            },
        ]
        result = format_thread_reply(sources)
        assert "sources:" in result
        assert "jake (2026-03-15)" in result
        assert "alice (2026-03-16)" in result
        assert "originals:" in result
        assert "<https://slack.com/p1|view>" in result
        assert "<https://slack.com/p2|view>" in result
        # Check middle dot separator
        assert "\u00b7" in result

    def test_no_permalinks(self):
        sources = [{"display_name": "jake", "date": "2026-03-15"}]
        result = format_thread_reply(sources)
        assert "source:" in result
        assert "jake (2026-03-15)" in result
        assert "original" not in result

    def test_custom_channel_name(self):
        sources = [{"display_name": "bob", "date": "2026-01-01"}]
        result = format_thread_reply(sources, channel_name="audio-dump")
        assert "#audio-dump" in result

    def test_no_source_info_in_main_comment(self):
        """Main comment should only have recipe info, not source attribution."""
        comment = format_main_comment("bone-xray")
        assert "source" not in comment.lower()
        assert "original" not in comment.lower()
        assert "permalink" not in comment.lower()


class TestResolveDisplayName:
    def test_returns_display_name(self):
        client = MagicMock()
        client.users_info.return_value = {
            "user": {
                "profile": {
                    "display_name": "Jake",
                    "real_name": "Jake Smith",
                }
            }
        }
        assert resolve_display_name(client, "U123") == "Jake"

    def test_falls_back_to_real_name(self):
        client = MagicMock()
        client.users_info.return_value = {
            "user": {
                "profile": {
                    "display_name": "",
                    "real_name": "Jake Smith",
                }
            }
        }
        assert resolve_display_name(client, "U123") == "Jake Smith"

    def test_falls_back_to_user_id(self):
        client = MagicMock()
        client.users_info.side_effect = Exception("API error")
        assert resolve_display_name(client, "U123") == "U123"


class TestTsToDate:
    def test_valid_ts(self):
        # 1700000000 = 2023-11-14
        assert _ts_to_date("1700000000.000100") == "2023-11-14"

    def test_invalid_ts(self):
        assert _ts_to_date("not-a-ts") == "unknown"

    def test_empty_ts(self):
        assert _ts_to_date("") == "unknown"
