#!/usr/bin/env python3
"""This script downloads the transcript of a YouTube video, using yt-dlp.

Usage:
    python3 scripts/download_transcript.py <url_youtube> [--lang es,en]
    python3 scripts/download_transcript.py [--ingestion-date 20260906]
    python3 scripts/download_transcript.py --clean [--ingestion-date 20260906]

Each video's ingestion date always comes from its own YouTube metadata field, `upload_date`
(format YYYYMMDD). You never pick this date by hand. A video uploaded on 2026-09-06 is saved
under `input/transcriptions/ingestion_date=20260906/`. Two videos downloaded in the same batch,
but uploaded on different dates, end up in different partitions. This is why `--ingestion-date`
is not accepted together with a URL: there is nothing to decide in that case.

`config.json` and `links.json` live under `input/`. They are global, not split by date. When
you run the script with no URL, it looks for `input/links.json` and re-downloads the
transcript for every link already in it. This is useful for refreshing all transcripts at
once, for example after you change the language in `config.json`. In this mode, and in
`--clean`, the flag `--ingestion-date <YYYYMMDD>` limits the operation to only the videos
whose `upload_date` matches. If you omit the flag, the script acts on all of them. A failure
on one video, such as a deleted video or missing subtitles in that language, does not stop the
rest of the batch.

The script picks which language to download in this order of priority: 1) `--lang`, if you
pass it explicitly; 2) the `"lang"` field in `input/config.json`, if it exists (for example
`{"lang": "en"}`); 3) `"es,en"` as the default. It tries the resulting languages in order, and
keeps the first one available, preferring manual subtitles over automatic ones. It stores
`{link, title, transcript_path, upload_date, description, channel_url}` in `input/links.json`,
adding a new entry or updating the existing one for each video.

Each download creates its own file, marked with the UTC timestamp of the moment of the
download (`<slug>.<lang>.<timestamp>.vtt`). Running the script again with the same URL never
overwrites or deletes a previous file. It always creates a new one. This also stops two
different videos whose titles produce the same slug from colliding under the same file name.
`links.json` still keeps only one entry per URL, though, updated to the latest download: the
title, language, and `transcript_path` of the most recent download. The history of `.vtt`
files lives only on disk, not in `links.json`.

`--clean` deletes the downloaded transcripts and their entries in `links.json`. Without
`--ingestion-date`, it deletes everything. With `--ingestion-date=<YYYYMMDD>`, it deletes only
that date.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone

try:
    import yt_dlp
except ImportError:
    sys.exit(
        "Install >> uv add yt-dlp  (this project's .venv is uv-managed and has no pip)"
    )

INPUT_ROOT = "input"
TRANSCRIPTIONS_ROOT = os.path.join(INPUT_ROOT, "transcriptions")
LOCK_FILE = ".download_transcript.pid"
SCRIPT_NAME = os.path.basename(__file__)

YDL_EXTRACTOR_ARGS = {"youtube": {"player_client": ["android"]}}


def _ingestion_date_dir(ingestion_date: str) -> str:
    return os.path.join(TRANSCRIPTIONS_ROOT, f"ingestion_date={ingestion_date}")


def _links_file() -> str:
    return os.path.join(INPUT_ROOT, "links.json")


def _config_file() -> str:
    return os.path.join(INPUT_ROOT, "config.json")


def _load_config() -> dict:
    path = _config_file()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _matching_keys(source: dict, lang: str) -> list[str]:
    """Matches a language. For example, 'es' matches 'es-419' and 'es-ES'."""
    exact = [lang] if lang in source else []
    variants = sorted(k for k in source if k != lang and k.split("-")[0] == lang)
    return exact + variants


def _pick_subtitle(info: dict, langs: list[str]) -> tuple[str, dict] | None:
    """Returns the subtitle language and format."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for lang in langs:
        for source in (manual, auto):
            for key in _matching_keys(source, lang):
                vtt = next((f for f in source[key] if f.get("ext") == "vtt"), None)
                if vtt:
                    return key, vtt
    return None


def _slugify(title: str) -> str:
    """Turns a title into a slug."""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_title).strip("_").lower()
    return slug or "untitled"


def _download_timestamp() -> str:
    """Returns the current UTC timestamp, with microsecond precision. This gives every
    downloaded transcript its own file name. So re-downloading the same URL never overwrites
    a previous snapshot, and two different videos that slugify the same way never collide."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"


def _timestamped_transcript_path(slug: str, lang: str, transcriptions_dir: str) -> str:
    """Returns the path `<transcriptions_dir>/<slug>.<lang>.<download_timestamp>.vtt`."""
    return os.path.join(transcriptions_dir, f"{slug}.{lang}.{_download_timestamp()}.vtt")


def _load_links(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _upsert_link(records: list[dict], record: dict) -> list[dict]:
    for i, existing in enumerate(records):
        if existing.get("link") == record["link"]:
            records[i] = record
            return records
    records.append(record)
    return records


def _save_links(path: str, records: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Temporary path.
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _clean_transcriptions(dir_path: str) -> int:
    """Removes every .vtt file under `dir_path`. Returns how many files it removed."""
    if not os.path.isdir(dir_path):
        return 0
    removed = 0
    for name in os.listdir(dir_path):
        if name.endswith(".vtt"):
            os.remove(os.path.join(dir_path, name))
            removed += 1
    return removed


def _clean(ingestion_date: str | None) -> tuple[int, int]:
    """Removes transcripts and their links.json entries. Returns (files_removed, records_removed)."""
    links_file = _links_file()
    records = _load_links(links_file)

    if ingestion_date:
        to_remove = [r for r in records if r.get("upload_date") == ingestion_date]
        keep = [r for r in records if r.get("upload_date") != ingestion_date]
        removed_files = _clean_transcriptions(_ingestion_date_dir(ingestion_date))
        _save_links(links_file, keep)
        return removed_files, len(to_remove)

    removed_files = 0
    if os.path.isdir(TRANSCRIPTIONS_ROOT):
        for entry in os.listdir(TRANSCRIPTIONS_ROOT):
            full = os.path.join(TRANSCRIPTIONS_ROOT, entry)
            if os.path.isdir(full):
                removed_files += _clean_transcriptions(full)
    removed_records = len(records)
    _save_links(links_file, [])
    return removed_files, removed_records


def _is_stale_script_process(pid: int) -> bool:
    """Checks whether 'pid' is still an active process."""
    try:
        out = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except OSError:
        return False
    return SCRIPT_NAME in out.stdout


def _kill_previous_instance(lock_path: str) -> None:
    if not os.path.exists(lock_path):
        return
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (ValueError, OSError):
        return
    if pid == os.getpid() or not _is_stale_script_process(pid):
        return
    try:
        os.kill(pid, signal.SIGKILL)
        print(f"Killed (PID {pid}) ", file=sys.stderr)
    except ProcessLookupError:
        pass


def _write_lock(lock_path: str) -> None:
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _download_url(url: str, dest: str, ydl: "yt_dlp.YoutubeDL", retries: int = 5) -> None:
    delay = 5
    for attempt in range(1, retries + 1):
        try:
            data = ydl.urlopen(url).read()
            with open(dest, "wb") as f:
                f.write(data)
            return
        except yt_dlp.networking.exceptions.HTTPError as e:
            if e.status != 429 or attempt == retries:
                raise
            retry_after = e.response.get_header("Retry-After")
            wait = int(retry_after) if retry_after and retry_after.isdigit() else delay
            print(
                f"YouTube devolvió 429 (Too Many Requests). Reintentando en {wait}s... ({attempt}/{retries})",
                file=sys.stderr,
            )
            time.sleep(wait)
            delay = min(delay * 2, 60)


def download_transcript(url: str, langs: list[str]) -> dict:
    links_file = _links_file()
    records = _load_links(links_file)

    ydl_opts = {
        "skip_download": True,
        "quiet": True,
        "no_warnings": True,
        "extractor_args": YDL_EXTRACTOR_ARGS,
        "ignore_no_formats_error": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

        video_id = info["id"]
        title = info.get("title", video_id)

        upload_date = info.get("upload_date")
        if not upload_date:
            raise RuntimeError(
                f"YouTube no reportó 'upload_date' para '{title}'; no se puede derivar su ingestion_date."
            )
        transcriptions_dir = _ingestion_date_dir(upload_date)
        os.makedirs(transcriptions_dir, exist_ok=True)

        pick = _pick_subtitle(info, langs)
        if pick is None:
            raise RuntimeError(
                f"No hay transcripción disponible (manual ni automática) para "
                f"los idiomas {langs} en '{title}'."
            )
        lang, fmt = pick

        # Every download gets its own file, marked with a timestamp. Re-downloading the same
        # URL never overwrites or removes a previous snapshot. It just adds a new one.
        slug = _slugify(title)
        transcript_path = _timestamped_transcript_path(slug, lang, transcriptions_dir)
        _download_url(fmt["url"], transcript_path, ydl)

    record = {
        "link": url,
        "title": title,
        "transcript_path": transcript_path,
        "lang": lang,
        "upload_date": upload_date,
        "description": info.get("description"),
        "channel_url": info.get("channel_url"),
    }

    records = _upsert_link(records, record)
    _save_links(links_file, records)

    return record


def download_all(langs: list[str], ingestion_date: str | None = None) -> list[dict]:
    """Downloads or re-downloads links tracked in input/links.json. You can filter this by
    `upload_date`.

    A failure on one video is reported and skipped. It does not stop the whole batch.
    Returns one summary dict per link: {"link", "ok", "result" | "error"}.
    """
    records = _load_links(_links_file())
    if ingestion_date:
        records = [r for r in records if r.get("upload_date") == ingestion_date]

    outcomes: list[dict] = []
    for record in records:
        url = record.get("link")
        if not url:
            continue
        try:
            result = download_transcript(url, langs)
            outcomes.append({"link": url, "ok": True, "result": result})
        except Exception as exc:  # Keep going. One bad video should not kill the whole batch.
            outcomes.append({"link": url, "ok": False, "error": str(exc)})
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Download transcription")
    parser.add_argument("url", nargs="?", help="Video link")
    parser.add_argument(
        "--ingestion-date",
        default=None,
        help=(
            "Filter by ingestion date (YYYYMMDD, matching videos' own YouTube upload_date). "
            "Only applies to bulk re-download or --clean; not accepted together with a "
            "single <url_youtube>, whose ingestion date is always derived from its own upload_date."
        ),
    )
    parser.add_argument(
        "--lang",
        default=None,
        help=(
            "Languages to try, comma-separated. Defaults to input/config.json's "
            "\"lang\" field if present, otherwise 'es,en'."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove downloaded transcripts and their links.json entries (see --ingestion-date).",
    )
    args = parser.parse_args()

    if args.url and args.ingestion_date:
        sys.exit(
            "--ingestion-date no se acepta junto a una URL: la fecha de ingesta se deriva del "
            "upload_date propio de ese vídeo, no hay nada que elegir."
        )

    if args.lang is not None:
        lang_str = args.lang
    else:
        config = _load_config()
        lang_str = config.get("lang")
        if lang_str:
            print(f"Usando idioma de {_config_file()}: {lang_str}", file=sys.stderr)
        else:
            lang_str = "es,en"
    langs = [lang.strip() for lang in lang_str.split(",") if lang.strip()]

    _kill_previous_instance(LOCK_FILE)
    _write_lock(LOCK_FILE)

    if args.clean:
        removed_files, removed_records = _clean(args.ingestion_date)
        scope = f"ingestion_date={args.ingestion_date}" if args.ingestion_date else "todas las fechas"
        print(
            f"{_links_file()} actualizado ({removed_records} enlace(s) eliminado(s)). "
            f"Borrados {removed_files} fichero(s) de transcripción de {scope}."
        )
        return

    if args.url:
        try:
            result = download_transcript(args.url, langs)
        except RuntimeError as exc:
            sys.exit(str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # No URL: this is bulk mode. It refreshes links already tracked, optionally filtered by
    # ingestion date.
    links_file = _links_file()
    if not os.path.exists(links_file):
        sys.exit(f"{links_file} no existe. Descarga al menos un vídeo con <url_youtube> primero.")

    outcomes = download_all(langs, args.ingestion_date)
    if not outcomes:
        scope = f" para ingestion_date={args.ingestion_date}" if args.ingestion_date else ""
        sys.exit(f"{links_file} no contiene ningún enlace{scope} que descargar.")

    ok_count = 0
    for outcome in outcomes:
        if outcome["ok"]:
            ok_count += 1
            print(f"✅ {outcome['link']} -> {outcome['result']['transcript_path']}")
        else:
            print(f"❌ {outcome['link']}: {outcome['error']}")

    fail_count = len(outcomes) - ok_count
    print(f"\n{ok_count} descargado(s), {fail_count} fallido(s) de {len(outcomes)} enlace(s).")
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == "__main__":
    main()
