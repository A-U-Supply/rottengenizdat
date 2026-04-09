# Sample Sale Integration — Design Spec

## Overview

Add the ability to fetch random audio/video files from the Slack #sample-sale channel and use them as inputs in the rottengenizdat pipeline. Includes multi-input support for both local files and Slack-sourced samples, with multiple combination modes.

## Multi-Input Support

### Current State

Every command (`rave`, `chain`, `recipe run`) takes a single `input_file: Path` positional argument.

### New Behavior

Commands accept multiple positional `input_files` arguments. Local files and sample-sale samples are merged into one input list before the pipeline runs.

```
rotten rave input1.wav input2.wav -m percussion --mode concat -o out.wav
rotten recipe run fever-dream.toml input1.wav input2.wav --sample-sale-count 2 -o out/
```

### Combination Modes (`--mode`)

- **`splice`** (default for multi-input) — Chop all inputs into random-length segments (default 0.25s–4.0s), shuffle them together, reassemble into one AudioBuffer, run pipeline once, one output file.
- **`concat`** — Join all inputs end-to-end in order into one AudioBuffer, run pipeline once, one output file.
- **`independent`** — Run the pipeline separately per input, produce multiple output files.

When only one input is provided, no `--mode` is needed (it runs as today). When multiple inputs are provided and no `--mode` is specified, `splice` is used.

When `--mode independent` is used with `-o`, the output path is treated as a directory. Output files are named `001-<source-filename>.wav`, `002-<source-filename>.wav`, etc.

### Splice Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--splice-min` | 0.25 | Minimum segment duration in seconds |
| `--splice-max` | 4.0 | Maximum segment duration in seconds |

Future enhancement: intelligent splicing that respects word boundaries, onset detection, and silence detection. The architecture supports swapping in a smarter splicing strategy since it's just the function that decides where to cut.

## Slack Integration

### Configuration

**Config file** at `~/.config/rottengenizdat/config.toml`:

```toml
[slack]
token = "xoxb-..."
channel = "C0XXXXXXX"
```

**Token resolution order:**
1. Config file `[slack].token`
2. `SLACK_BOT_TOKEN` environment variable
3. Error with setup instructions

### Fetching Logic

Module: `src/rottengenizdat/sample_sale.py`

Uses the Slack `conversations.history` API to page through channel messages. Extracts two kinds of media:

- **File attachments** — Filtered by audio/video MIME types. Downloaded via Slack file URL with auth header.
- **Links** — URLs extracted from message text. Downloaded via `yt-dlp`. If yt-dlp is not installed or the URL is not supported, the link is indexed but skipped during download with a warning.

Non-audio/video files (images, PDFs, text snippets, etc.) are silently skipped during indexing.

### Cache

**Media cache directory:** `~/.cache/rottengenizdat/samples/`

Downloaded files are stored here, keyed by Slack file ID (for attachments) or URL hash (for links). Files are not re-downloaded if already cached.

**Cache index:** `~/.cache/rottengenizdat/index.json`

```json
[
  {
    "id": "F07XXXX",
    "type": "attachment",
    "filename": "loop.wav",
    "mime": "audio/wav",
    "slack_url": "https://files.slack.com/...",
    "cached_path": "samples/F07XXXX-loop.wav",
    "message_ts": "1712345678.000100"
  },
  {
    "id": "sha256-abcdef",
    "type": "link",
    "url": "https://youtube.com/...",
    "cached_path": "samples/sha256-abcdef.wav",
    "message_ts": "1712345679.000200"
  }
]
```

### Incremental Sync

On every `--sample-sale` use, the tool fetches messages newer than the latest `message_ts` in the index and appends new entries. Actual media files are downloaded lazily — only when a sample is randomly selected for use, not during indexing.

A `rotten sample-sale refresh` command also triggers the incremental sync explicitly.

## CLI Changes

### New Flags on Pipeline Commands

Added to `recipe run`, `chain`, and individual effect commands (e.g. `rave`):

| Flag | Description |
|------|-------------|
| `input_files` (positional, multiple) | Local audio/video files |
| `--sample-sale` / `-ss` | Include random samples from #sample-sale (implies count of 1) |
| `--sample-sale-count N` | How many samples to pull (implies `--sample-sale`) |
| `--mode` | `splice` (default for multi-input), `concat`, `independent` |
| `--splice-min` | Min segment duration in seconds (default 0.25) |
| `--splice-max` | Max segment duration in seconds (default 4.0) |

### `rotten sample-sale` Subcommand

```
rotten sample-sale refresh          # incremental sync (new messages since last sync)
rotten sample-sale refresh --full   # rebuild index from scratch
rotten sample-sale list             # show all indexed samples (id, filename, type, cached?)
rotten sample-sale list --cached    # only show already-downloaded samples
rotten sample-sale clear            # delete cached media files (keeps index)
rotten sample-sale clear --all      # delete cache dir entirely (media + index)
```

### `rotten config` Subcommand

```
rotten config show                  # print current config (token value masked)
rotten config set slack.token xoxb-...
rotten config set slack.channel C0XXXXXXX
rotten config path                  # print config file location
```

Writes to `~/.config/rottengenizdat/config.toml`. Creates the file and parent directories if they don't exist.

## New Dependencies

- **`slack_sdk`** — Slack API client for channel history and file downloads.
- **`yt-dlp`** — Optional runtime dependency for resolving media links. Detected at runtime; if missing, links are indexed but skipped during download with a warning suggesting installation.

## New Modules

| Module | Responsibility |
|--------|---------------|
| `src/rottengenizdat/sample_sale.py` | Slack API interaction, index management, media downloading, random sample selection |
| `src/rottengenizdat/config.py` | Config file read/write, token resolution, config CLI commands |
| `src/rottengenizdat/splice.py` | Splice mode implementation (chop, shuffle, reassemble AudioBuffers) |
| `src/rottengenizdat/inputs.py` | Multi-input resolution (merge local files + sample-sale picks into input list), combination mode dispatch |

## Testing

- Unit tests for splice logic (deterministic with seeded RNG)
- Unit tests for config resolution (file priority over env var)
- Unit tests for index incremental sync logic (mock Slack API)
- Unit tests for multi-input mode dispatch
- Integration test for full pipeline with multiple local inputs
