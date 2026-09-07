#!/usr/bin/env python3
"""Descarga la transcripción de un vídeo de YouTube con yt-dlp.

Uso:
    python3 scripts/download_transcript.py <url_youtube> [--lang es,en]
    python3 scripts/download_transcript.py [--ingestion-date 20260906]
    python3 scripts/download_transcript.py --clean [--ingestion-date 20260906]

La fecha de ingesta de cada vídeo se deriva siempre de su propio metadato de
YouTube `upload_date` (formato YYYYMMDD) — no se elige a mano. Un vídeo
subido el 2026-09-06 se guarda en
`input/transcriptions/ingestion_date=20260906/`; dos vídeos descargados en el
mismo lote pero subidos en fechas distintas terminan en particiones
distintas. `--ingestion-date` no se acepta junto a una URL por esa razón: no
hay nada que decidir.

`config.json` y `links.json` viven en `input/` (globales, no por fecha).
Sin URL, el script busca `input/links.json` y (re)descarga la transcripción
de cada enlace que ya contiene — útil para refrescar todas las
transcripciones de golpe (p.ej. tras cambiar el idioma en config.json). En
este modo (y en `--clean`), `--ingestion-date <YYYYMMDD>` filtra la operación
a solo los vídeos cuyo `upload_date` coincida; omitido, actúa sobre todos. Un
fallo en un vídeo (borrado, sin subtítulos en ese idioma...) no aborta el
resto del lote.

El idioma a descargar se resuelve en este orden de prioridad: 1) --lang si se
pasa explícitamente, 2) el campo "lang" de input/config.json si existe (p.ej.
{"lang": "en"}), 3) "es,en" por defecto. Prueba los idiomas resultantes en
orden y se queda con el primero disponible (subtítulos manuales antes que
automáticos). Acumula {link, title, transcript_path, upload_date,
description, channel_url} en input/links.json, añadiendo o actualizando la
entrada de cada vídeo.

Es idempotente: volver a ejecutarlo con la misma URL sobrescribe el mismo
fichero .vtt y actualiza la misma entrada de links.json, sin acumular
ficheros nuevos, aunque el título del vídeo haya cambiado entre ejecuciones.
Si el idioma resuelto cambia (p.ej. cambiaste el "lang" de config.json), el
nombre del fichero se recalcula para reflejar el nuevo idioma y el fichero
antiguo se borra — nunca deja un `*.es.vtt` con contenido en otro idioma
dentro.

--clean borra las transcripciones descargadas y sus entradas de links.json;
sin --ingestion-date, todo; con --ingestion-date=<YYYYMMDD>, solo esa fecha.
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
    """Language match (ex. 'es' matchs 'es-419', 'es-ES')."""
    exact = [lang] if lang in source else []
    variants = sorted(k for k in source if k != lang and k.split("-")[0] == lang)
    return exact + variants


def _pick_subtitle(info: dict, langs: list[str]) -> tuple[str, dict] | None:
    """Return subtitle lang and format"""
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
    """title slug"""
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", ascii_title).strip("_").lower()
    return slug or "untitled"


def _unique_transcript_path(
    slug: str, lang: str, url: str, records: list[dict], transcriptions_dir: str
) -> str:
    """Path `<transcriptions_dir>/<slug>.<lang>.vtt`"""
    suffix = ""
    n = 2
    while True:
        candidate = os.path.join(transcriptions_dir, f"{slug}{suffix}.{lang}.vtt")
        clash = any(
            r.get("transcript_path") == candidate and r.get("link") != url
            for r in records
        )
        if not clash:
            return candidate
        suffix = f"_{n}"
        n += 1


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
    # Temporary path
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _clean_transcriptions(dir_path: str) -> int:
    """Remove every .vtt file under `dir_path`. Returns how many were removed."""
    if not os.path.isdir(dir_path):
        return 0
    removed = 0
    for name in os.listdir(dir_path):
        if name.endswith(".vtt"):
            os.remove(os.path.join(dir_path, name))
            removed += 1
    return removed


def _clean(ingestion_date: str | None) -> tuple[int, int]:
    """Remove transcripts and their links.json entries. Returns (files_removed, records_removed)."""
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
    """check active 'pid'"""
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
    existing = next((r for r in records if r.get("link") == url), None)

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

        # Re-run for the same URL reuses its existing path, so the file is
        # overwritten in place instead of accumulating a new one (idempotent
        # even if the video's title, and therefore its slug, has changed) —
        # but only when the resolved language is unchanged. If it changed
        # (e.g. config.json now asks for a different language), the filename
        # must change with it: reusing the old path would leave a file whose
        # name still claims the old language while its content is actually
        # in the new one. Compute a fresh, language-correct path instead, and
        # remove the now-stale file under the old name.
        old_path = existing.get("transcript_path") if existing else None
        old_lang = existing.get("lang") if existing else None
        if old_path and old_lang == lang:
            transcript_path = old_path
        else:
            slug = _slugify(title)
            transcript_path = _unique_transcript_path(slug, lang, url, records, transcriptions_dir)
            if old_path and old_path != transcript_path and os.path.exists(old_path):
                os.remove(old_path)
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
    """(Re)download links tracked in input/links.json, optionally filtered by `upload_date`.

    A failure on one video is reported and skipped, not fatal to the batch.
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
        except Exception as exc:  # keep going, one bad video shouldn't kill the batch
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

    # No URL: bulk mode — refresh links already tracked, optionally filtered by ingestion date.
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
