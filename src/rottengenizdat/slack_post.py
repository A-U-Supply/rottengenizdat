"""Post processed audio to Slack with source attribution in thread replies.

Follows the same battle-tested pattern as sparagmos: upload the file,
aggressively retry to find the message timestamp (Slack's eventual
consistency), then post source attribution as a thread reply.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from slack_sdk import WebClient

logger = logging.getLogger(__name__)


def format_main_comment(label: str, run_url: str = "") -> str:
    """Format the main Slack message: recipe label + optional logs link."""
    lines = [f":radio: *rottengenizdat: {label}*"]
    if run_url:
        lines.append(f"<{run_url}|View logs>")
    return "\n".join(lines)


def resolve_display_name(client: WebClient, user_id: str) -> str:
    """Resolve a Slack user ID to a plain display name.

    Falls back to real_name, then the raw user_id on failure.
    """
    try:
        resp = client.users_info(user=user_id)
        profile = resp["user"]["profile"]
        return profile.get("display_name") or profile.get("real_name") or user_id
    except Exception:
        logger.warning("Failed to resolve display name for %s", user_id)
        return user_id


def _get_permalink(client: WebClient, channel: str, message_ts: str) -> str:
    """Get a permalink for a Slack message."""
    try:
        resp = client.chat_getPermalink(channel=channel, message_ts=message_ts)
        return resp.get("permalink", "")
    except Exception:
        logger.warning("Failed to get permalink for ts=%s", message_ts)
        return ""


def format_thread_reply(
    sources: list[dict],
    channel_name: str = "sample-sale",
) -> str:
    """Format the thread reply with source attribution and permalink links.

    Handles two source types:
    - #sample-sale sources: show display_name, date, permalink
    - User-provided URLs: show the URL directly as a clickable link

    Args:
        sources: List of source dicts with 'display_name', 'date', and
            optional 'permalink' and 'url' keys.
        channel_name: Source channel name for attribution.

    Returns:
        Formatted string for the thread reply text.
    """
    channel_sources = [s for s in sources if not s.get("url")]
    url_sources = [s for s in sources if s.get("url")]

    lines: list[str] = []

    if channel_sources:
        source_label = "source" if len(channel_sources) == 1 else "sources"
        attributions = ", ".join(
            f"{s['display_name']} ({s.get('date', 'unknown')})" for s in channel_sources
        )
        lines.append(f"{source_label}: {attributions} in #{channel_name}")

        permalinks = [s.get("permalink", "") for s in channel_sources if s.get("permalink")]
        if permalinks:
            link_label = "original" if len(permalinks) == 1 else "originals"
            links = " \u00b7 ".join(f"<{url}|view>" for url in permalinks)
            lines.append(f"{link_label}: {links}")

    if url_sources:
        url_label = "input url" if len(url_sources) == 1 else "input urls"
        url_links = " \u00b7 ".join(f"<{s['url']}|link>" for s in url_sources)
        lines.append(f"{url_label}: {url_links}")

    return "\n".join(lines)


def _find_posted_ts(
    client: WebClient, channel_id: str, file_id: str
) -> str:
    """Find the message timestamp for an uploaded file.

    Uses the same aggressive retry strategy as sparagmos: up to 50
    attempts with exponential backoff, trying files.info first then
    falling back to conversations.history scan.

    Slack's eventual consistency means neither API reflects the share
    immediately after upload.
    """
    max_attempts = 50
    for attempt in range(max_attempts):
        # Exponential backoff: 1, 2, 4, 8 ... capped at 10s
        time.sleep(min(2**attempt, 10))

        # Primary: files.info → shares gives us ts directly
        try:
            info = client.files_info(file=file_id)
            shares = info.get("file", {}).get("shares", {})
            for visibility in ("public", "private"):
                channel_shares = shares.get(visibility, {}).get(channel_id, [])
                if channel_shares:
                    posted_ts = channel_shares[0].get("ts", "")
                    if posted_ts:
                        logger.info(
                            "Found posted_ts via files.info on attempt %d",
                            attempt + 1,
                        )
                        return posted_ts
        except Exception:
            pass

        # Fallback: scan recent channel history for our file ID
        try:
            history = client.conversations_history(
                channel=channel_id, limit=10
            )
            for msg in history.get("messages", []):
                if any(
                    f.get("id") == file_id for f in msg.get("files", [])
                ):
                    posted_ts = msg["ts"]
                    logger.info(
                        "Found posted_ts via conversations.history on attempt %d",
                        attempt + 1,
                    )
                    return posted_ts
        except Exception:
            pass

        logger.debug(
            "Attempt %d/%d: file %s not yet visible in channel",
            attempt + 1,
            max_attempts,
            file_id,
        )

    logger.error(
        "Could not find uploaded message after %d attempts (file_id=%s)",
        max_attempts,
        file_id,
    )
    return ""


def _ts_to_date(ts: str) -> str:
    """Convert a Slack message timestamp to a YYYY-MM-DD date string."""
    try:
        epoch = float(ts.split(".")[0])
        return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")
    except (ValueError, IndexError):
        return "unknown"


def post_result(
    client: WebClient,
    channel_id: str,
    audio_path: Path,
    label: str,
    sources: list[dict],
    source_channel_id: str,
    source_channel_name: str = "sample-sale",
    run_url: str = "",
) -> str:
    """Post a processed audio file to Slack with source info in a thread reply.

    Uploads the output audio with a main comment (recipe label + logs link),
    then posts source attribution and permalink links as a thread reply.

    Args:
        client: Slack WebClient.
        channel_id: Target channel ID for posting.
        audio_path: Path to the output audio file.
        label: Recipe/chain label for the main comment.
        sources: List of source metadata dicts from sources.json.
        source_channel_id: Channel ID where samples came from (for permalinks).
        source_channel_name: Display name of source channel.
        run_url: Optional URL to GitHub Actions run logs.

    Returns:
        Message timestamp of the posted message.
    """
    comment = format_main_comment(label, run_url)
    logger.info("Posting to channel %s with comment:\n%s", channel_id, comment)

    # Upload the audio file
    response = client.files_upload_v2(
        channel=channel_id,
        file=str(audio_path),
        filename=f"rottengenizdat-{label}.wav",
        initial_comment=comment,
    )

    # Extract file ID from upload response
    file_obj = response.get("file") or {}
    if not file_obj:
        files_list = response.get("files") or []
        if files_list:
            file_obj = files_list[0]
    file_id = file_obj.get("id", "")

    if not file_id:
        logger.warning("No file ID from upload response, skipping thread reply")
        return ""

    # Find the message containing our uploaded file
    posted_ts = _find_posted_ts(client, channel_id, file_id)

    # Post source attribution as a thread reply
    if posted_ts and sources:
        # Resolve display names and get permalinks for each source
        resolved_sources = []
        for src in sources:
            display_name = src.get("user", "unknown")
            if display_name and display_name != "unknown":
                display_name = resolve_display_name(client, display_name)

            permalink = ""
            if src.get("message_ts") and source_channel_id:
                permalink = _get_permalink(
                    client, source_channel_id, src["message_ts"]
                )

            resolved_sources.append(
                {
                    "display_name": display_name,
                    "date": _ts_to_date(src.get("message_ts", "")),
                    "permalink": permalink,
                    "url": src.get("url", ""),
                }
            )

        thread_text = format_thread_reply(resolved_sources, source_channel_name)
        try:
            client.chat_postMessage(
                channel=channel_id,
                thread_ts=posted_ts,
                text=thread_text,
            )
            logger.info("Posted thread reply with source attribution")
        except Exception:
            logger.warning("Failed to post thread reply", exc_info=True)

    return posted_ts


def post_from_sources_file(
    token: str,
    channel_id: str,
    audio_path: Path,
    label: str,
    sources_file: Path | None = None,
    source_channel_id: str = "",
    source_channel_name: str = "sample-sale",
    run_url: str = "",
) -> str:
    """Convenience wrapper that reads sources from a JSON file.

    Args:
        token: Slack bot token.
        channel_id: Target channel ID for posting.
        audio_path: Path to the output audio file.
        label: Recipe/chain label.
        sources_file: Path to sources.json (optional).
        source_channel_id: Channel ID where samples came from.
        source_channel_name: Display name of source channel.
        run_url: Optional URL to GitHub Actions run logs.

    Returns:
        Message timestamp of the posted message.
    """
    client = WebClient(token=token)

    sources: list[dict] = []
    if sources_file and sources_file.exists():
        sources = json.loads(sources_file.read_text())

    return post_result(
        client=client,
        channel_id=channel_id,
        audio_path=audio_path,
        label=label,
        sources=sources,
        source_channel_id=source_channel_id,
        source_channel_name=source_channel_name,
        run_url=run_url,
    )
