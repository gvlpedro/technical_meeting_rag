#!/usr/bin/env python3
"""Valida los data contracts ODCS generados por prompting/roles/common/data_contracts.md.

Uso:
    python3 scripts/validate_data_contracts.py --session session_20260906
    python3 scripts/validate_data_contracts.py --session session_20260906 --dir otro_output
    python3 scripts/validate_data_contracts.py archivo1.odcs.json archivo2.odcs.json

`--session` es obligatorio salvo que pases ficheros concretos como argumentos:
busca todos los `*.odcs.json` bajo `<dir>/<session>/data_contracts/`
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DATACONTRACT_BIN = "datacontract"
DEFAULT_ROOT = "output"
CONTRACT_GLOB = "*.odcs.json"


def find_contracts(root: str, session: str) -> list[Path]:
    """Only look under <root>/<session>/data_contracts/ — never other sessions."""
    base = Path(root) / session / "data_contracts"
    return sorted(base.rglob(CONTRACT_GLOB))


def lint_one(path: Path) -> tuple[bool, list[str]]:
    """Run `datacontract lint` on a single contract file.

    Returns (is_valid, reasons). `reasons` is empty when valid.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        proc = subprocess.run(
            [
                DATACONTRACT_BIN,
                "lint",
                "--all-errors",
                "--output-format",
                "json",
                "--output",
                str(tmp_path),
                str(path),
            ],
            capture_output=True,
            text=True,
        )
        is_valid = proc.returncode == 0

        reasons: list[str] = []
        try:
            report = json.loads(tmp_path.read_text(encoding="utf-8"))
            for check in report.get("checks", []):
                if check.get("result") == "failed":
                    reasons.append(check.get("reason") or check.get("name") or "unknown error")
        except (json.JSONDecodeError, OSError):
            pass

        if not is_valid and not reasons:
            # Fall back to whatever the CLI printed if the JSON report is unavailable.
            reasons = [
                line
                for line in (proc.stdout + proc.stderr).splitlines()
                if line.strip()
            ] or ["datacontract lint failed with no captured output"]

        return is_valid, reasons
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ODCS data contracts")
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific *.odcs.json files to validate. If omitted, auto-discovers them.",
    )
    parser.add_argument(
        "--dir",
        default=DEFAULT_ROOT,
        help=f"Root directory to search under (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--session",
        default=None,
        help=(
            "Session to validate (e.g. session_20260906). Only contracts under "
            "<dir>/<session>/data_contracts/ are looked at. Mandatory unless "
            "FILES are given explicitly."
        ),
    )
    args = parser.parse_args()

    if not args.files and not args.session:
        parser.error("--session is required when no explicit files are given")

    if shutil.which(DATACONTRACT_BIN) is None:
        sys.exit(
            f"'{DATACONTRACT_BIN}' no está instalado o no está en el PATH.\n"
            "Instálalo con: uv tool install datacontract-cli"
        )

    contracts = [Path(f) for f in args.files] if args.files else find_contracts(args.dir, args.session)

    if not contracts:
        print("No se encontró ningún *.odcs.json para validar.")
        sys.exit(1)

    results: list[tuple[Path, bool, list[str]]] = []
    for path in contracts:
        is_valid, reasons = lint_one(path)
        results.append((path, is_valid, reasons))

    valid_count = sum(1 for _, ok, _ in results if ok)
    invalid_count = len(results) - valid_count

    for path, is_valid, reasons in results:
        mark = "✅" if is_valid else "❌"
        print(f"{mark} {path}")
        for reason in reasons:
            print(f"    - {reason}")

    print(f"\n{valid_count} válido(s), {invalid_count} inválido(s) de {len(results)} contrato(s).")

    sys.exit(0 if invalid_count == 0 else 1)


if __name__ == "__main__":
    main()
