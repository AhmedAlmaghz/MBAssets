"""Quality gate for the MaghzBook bundle's internal links.

Checks:
  * no host-bound asset URLs remain (raw.githubusercontent / github.io)
  * every HTML has <meta charset> in the first 1024 bytes + the APP_URL resolver
  * every data-mb-path, relative href/src and JSON internal path resolves to a
    real file inside the bundle
  * manifest.json sizes and SHA-256 checksums still match the files
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "MaghzBook", HERE.parent]
TEXT_EXT = {".html", ".json", ".md", ".js", ".css", ".txt"}
DIRS = r"(?:books|courses|landing|brand|tools)"
PCHAR = r"[A-Za-z0-9_./%\-]"
ROOT_FILES = r"(?:index\.html|catalog\.json|manifest\.json|README\.md|\.env\.example)"

HOST_RE = re.compile(
    r"raw\.githubusercontent\.com/AhmedAlmaghz/MBAssets|ahmedalmaghz\.github\.io",
    re.I,
)
MB_PATH_RE = re.compile(r'\bdata-mb-path="(' + PCHAR + r'+)"')
REL_RE = re.compile(r'\b(?:href|src|content|poster)="((?:\./|\.\./)+' + DIRS + r"/" + PCHAR + r'+)"')
EXACT_RE = re.compile(r"^(?:" + DIRS + r"/" + PCHAR + r"+|" + ROOT_FILES + r")$")
MD_LINK_RE = re.compile(r"\]\(((?:\./)?" + DIRS + r"/" + PCHAR + r"+)\)")


def find_bundle() -> Path:
    for p in CANDIDATES:
        if (p / "catalog.json").exists():
            return p
    raise SystemExit("MaghzBook bundle not found")


def main() -> int:
    root = find_bundle()
    errors: list[str] = []
    checked = 0
    files = [p for p in sorted(root.rglob("*"))
             if p.is_file() and ".git" not in p.parts and p.suffix.lower() in TEXT_EXT]

    for p in files:
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in HOST_RE.finditer(text):
            errors.append(f"host-bound URL in {p.relative_to(root)}: {m.group(0)}")

        if p.suffix.lower() == ".html":
            raw = p.read_bytes()
            if raw.find(b"<meta charset") == -1 or raw.find(b"<meta charset") > 1024:
                errors.append(f"charset missing/late in {p.name}")
            if b'name="app-url"' not in raw:
                errors.append(f"APP_URL resolver missing in {p.name}")
            for path in MB_PATH_RE.findall(text):
                checked += 1
                if not (root / path).exists():
                    errors.append(f"missing data-mb-path target {path} in {p.name}")
            for path in REL_RE.findall(text):
                checked += 1
                target = posixpath.normpath(
                    posixpath.join(p.parent.relative_to(root).as_posix() or ".", path)
                )
                if not (root / target).exists():
                    errors.append(f"missing relative link {path} -> {target} in {p.name}")

        elif p.suffix.lower() == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                errors.append(f"invalid JSON {p.name}: {e}")
                continue

            def walk(o):
                nonlocal checked
                if isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)
                elif isinstance(o, str):
                    if EXACT_RE.fullmatch(o):
                        checked += 1
                        if o not in ("",) and not (root / o).exists():
                            errors.append(f"missing JSON path {o} in {p.name}")
                    for link in MD_LINK_RE.findall(o):
                        checked += 1
                        target = link[2:] if link.startswith("./") else link
                        if not (root / target).exists():
                            errors.append(f"missing markdown link {link} in {p.name}")
            walk(data)

    manifest = root / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        for b in data.get("books", []):
            for key, rel in b.get("files", {}).items():
                f = root / rel
                if not f.is_file():
                    continue
                if key in b.get("sizes", {}) and b["sizes"][key] != f.stat().st_size:
                    errors.append(f"stale size {b['slug']}.{key}")
                if key in b.get("sha256", {}):
                    digest = hashlib.sha256(f.read_bytes()).hexdigest()
                    if digest != b["sha256"][key]:
                        errors.append(f"stale sha256 {b['slug']}.{key}")

    print(f"bundle        : {root}")
    print(f"text files    : {len(files)}")
    print(f"links checked : {checked}")
    print(f"problems      : {len(errors)}")
    for e in errors[:40]:
        print("  !", e)
    if len(errors) > 40:
        print(f"  ... and {len(errors) - 40} more")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
