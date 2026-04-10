from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from rottengenizdat.config import resolve_slack_channel, resolve_slack_token

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "rottengenizdat"
INDEX_FILENAME = "index.json"
SAMPLES_DIRNAME = "samples"

_MEDIA_PREFIXES = ("audio/", "video/")
_URL_PATTERN = re.compile(r"https?://[^\s<>|]+")


@dataclass
class IndexEntry:
    """One indexed media item from #sample-sale."""

    id: str
    type: str
    message_ts: str
    user: str = ""
    filename: str = ""
    mime: str = ""
    slack_url: str = ""
    url: str = ""
    cached_path: str = ""

    @property
    def is_cached(self) -> bool:
        if not self.cached_path:
            return False
        return (CACHE_DIR / self.cached_path).exists()


def _index_path(cache_dir: Path = CACHE_DIR) -> Path:
    return cache_dir / INDEX_FILENAME


def load_index(cache_dir: Path = CACHE_DIR) -> list[IndexEntry]:
    """Load the sample index from disk."""
    path = _index_path(cache_dir)
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)
    return [IndexEntry(**entry) for entry in data]


def save_index(entries: list[IndexEntry], cache_dir: Path = CACHE_DIR) -> None:
    """Write the sample index to disk."""
    path = _index_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump([asdict(e) for e in entries], f, indent=2)


def _url_hash(url: str) -> str:
    """Deterministic short hash for a URL."""
    return "sha256-" + hashlib.sha256(url.encode()).hexdigest()[:12]


def extract_media_from_messages(messages: list[dict]) -> list[IndexEntry]:
    """Extract audio/video entries from Slack messages."""
    seen_ids: set[str] = set()
    entries: list[IndexEntry] = []

    for msg in messages:
        ts = msg.get("ts", "")
        msg_user = msg.get("user", "")

        for file_info in msg.get("files", []):
            mime = file_info.get("mimetype", "")
            if not any(mime.startswith(p) for p in _MEDIA_PREFIXES):
                continue
            file_id = file_info["id"]
            if file_id in seen_ids:
                continue
            seen_ids.add(file_id)
            filename = file_info.get("name", "unknown")
            entries.append(
                IndexEntry(
                    id=file_id,
                    type="attachment",
                    filename=filename,
                    mime=mime,
                    slack_url=file_info.get("url_private_download", ""),
                    cached_path=f"{SAMPLES_DIRNAME}/{file_id}-{filename}",
                    message_ts=ts,
                    user=msg_user,
                )
            )

        text = msg.get("text", "")
        for url in _URL_PATTERN.findall(text):
            entry_id = _url_hash(url)
            if entry_id in seen_ids:
                continue
            seen_ids.add(entry_id)
            entries.append(
                IndexEntry(
                    id=entry_id,
                    type="link",
                    url=url,
                    cached_path=f"{SAMPLES_DIRNAME}/{entry_id}.wav",
                    message_ts=ts,
                    user=msg_user,
                )
            )

    return entries


def sync_index(
    full: bool = False,
    cache_dir: Path = CACHE_DIR,
    config_dir: Path | None = None,
) -> list[IndexEntry]:
    """Fetch new messages from Slack and update the index."""
    from slack_sdk import WebClient

    kwargs = {"config_dir": config_dir} if config_dir else {}
    token = resolve_slack_token(**kwargs)
    channel = resolve_slack_channel(**kwargs)
    client = WebClient(token=token)

    existing = [] if full else load_index(cache_dir)
    latest_ts = "0"
    if existing and not full:
        latest_ts = max(e.message_ts for e in existing)

    new_messages: list[dict] = []
    cursor = None
    while True:
        resp = client.conversations_history(
            channel=channel,
            oldest=latest_ts,
            limit=200,
            cursor=cursor,
        )
        new_messages.extend(resp["messages"])
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    new_entries = extract_media_from_messages(new_messages)

    existing_ids = {e.id for e in existing}
    unique_new = [e for e in new_entries if e.id not in existing_ids]

    combined = existing + unique_new
    save_index(combined, cache_dir)
    return combined


def _download_slack_file(url: str, token: str) -> bytes:
    """Download a file from Slack, preserving auth through redirects.

    Follows up to 5 redirects with the Authorization header on every
    request (requests strips it on cross-host redirect by default).
    Matches the battle-tested pattern from sparagmos.
    """
    import requests

    headers = {"Authorization": f"Bearer {token}"}
    max_redirects = 5
    for _ in range(max_redirects):
        resp = requests.get(
            url, headers=headers, timeout=30, allow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            url = resp.headers["Location"]
            continue
        resp.raise_for_status()

        content_type = resp.headers.get("Content-Type", "")
        if content_type.startswith(("text/html", "application/json")):
            raise RuntimeError(
                f"Expected media, got {content_type}. "
                "Slack may have returned a login/error page."
            )
        return resp.content

    raise RuntimeError(f"Too many redirects downloading {url}")


def download_sample(
    entry: IndexEntry,
    cache_dir: Path = CACHE_DIR,
    config_dir: Path | None = None,
) -> Path:
    """Download a sample if not already cached. Returns the local file path."""
    local_path = cache_dir / entry.cached_path
    if local_path.exists() and local_path.stat().st_size > 1024:
        return local_path
    # Remove any cached corrupt/truncated files
    if local_path.exists():
        local_path.unlink()

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if entry.type == "attachment":
        kwargs = {"config_dir": config_dir} if config_dir else {}
        token = resolve_slack_token(**kwargs)

        # Get a fresh download URL via files.info — cached URLs can go
        # stale when files are re-shared or permissions change.
        from slack_sdk import WebClient

        client = WebClient(token=token)
        try:
            info = client.files_info(file=entry.id)
            file_data = info.get("file", {})
            fresh_url = (
                file_data.get("url_private_download")
                or file_data.get("url_private")
                or entry.slack_url
            )
        except Exception:
            logger.warning(
                "files.info failed for %s, using cached URL", entry.id
            )
            fresh_url = entry.slack_url

        if not fresh_url:
            raise RuntimeError(
                f"No download URL for {entry.filename or entry.id}"
            )

        data = _download_slack_file(fresh_url, token)
        local_path.write_bytes(data)
    elif entry.type == "link":
        if not shutil.which("yt-dlp"):
            raise RuntimeError(
                "yt-dlp is not installed. Install it to download linked media:\n"
                "  pip install yt-dlp"
            )
        # Use a temp directory so yt-dlp can name files however it wants,
        # then move the result to our expected cache path.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--extract-audio",
                    "--audio-format", "wav",
                    "--output", f"{tmp}/%(id)s.%(ext)s",
                    entry.url,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"yt-dlp failed for {entry.url}: {result.stderr}")
            wavs = list(Path(tmp).glob("*.wav"))
            if not wavs:
                raise RuntimeError(
                    f"yt-dlp produced no .wav file for {entry.url}"
                )
            shutil.move(str(wavs[0]), str(local_path))

    return local_path


def pick_random_samples(
    entries: list[IndexEntry], count: int
) -> list[IndexEntry]:
    """Pick N random samples from the index."""
    if not entries:
        raise ValueError("No samples in index. Run: rotten sample-sale refresh")
    return random.sample(entries, min(count, len(entries)))


def clear_cache(cache_dir: Path = CACHE_DIR, full: bool = False) -> None:
    """Delete cached sample files, optionally the entire cache dir."""
    samples_dir = cache_dir / SAMPLES_DIRNAME
    if samples_dir.exists():
        shutil.rmtree(samples_dir)
    if full:
        index = _index_path(cache_dir)
        if index.exists():
            index.unlink()
