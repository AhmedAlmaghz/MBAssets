"""Prepend APP_URL before every internal MaghzBook link (export / build step).

The committed bundle is intentionally host-agnostic: all links are relative.
When an app needs absolute URLs (separate asset origin, CDN, server-side
import of blueprints), set the APP_URL environment variable and run:

    APP_URL=https://books.example.com python tools/resolve_urls.py --export ../dist
    APP_URL=https://books.example.com python tools/resolve_urls.py --in-place

With APP_URL unset, links stay internal and the export is a plain copy.
Safe to re-run: absolute links are not prefixed twice.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "MaghzBook", HERE.parent]
TEXT_EXT = {".html", ".json", ".md", ".js", ".css", ".txt"}
DIRS = r"(?:books|courses|landing|brand|tools)"
PCHAR = r"[A-Za-z0-9_./%\-]"
ROOT_FILES = r"(?:index\.html|catalog\.json|manifest\.json|README\.md|\.env\.example)"
INNER = r"(?:" + DIRS + r"/" + PCHAR + r"+|" + ROOT_FILES + r")"

REL_RE = re.compile(r"(?:\./|\.\./)+(?:MaghzBook/)?" + INNER)
EXACT_RE = re.compile(r"^(?:MaghzBook/)?" + INNER + r"$")
MD_LINK_RE = re.compile(r"(\]\()((?:\./)?" + DIRS + r"/" + PCHAR + r"+)(\))")


def find_bundle(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).resolve()
        if not (p / "catalog.json").exists():
            raise SystemExit(f"Not a MaghzBook bundle: {p}")
        return p
    for p in CANDIDATES:
        if (p / "catalog.json").exists():
            return p
    raise SystemExit("MaghzBook bundle not found; pass --bundle DIR")


def app_url() -> str:
    return os.environ.get("APP_URL", "").strip().rstrip("/")


def strip_dot(path: str) -> str:
    while path.startswith("./"):
        path = path[2:]
    return path


def canonical(doc: Path, root: Path, rel: str) -> str:
    here = doc.parent.relative_to(root).as_posix()
    return strip_dot(posixpath.normpath(posixpath.join(here or ".", rel)))


def set_meta(html: str, base: str) -> str:
    if re.search(r'<meta name="app-url"', html):
        return re.sub(r'(<meta name="app-url" content=")[^"]*(")',
                      lambda m: m.group(1) + base + m.group(2), html, count=1)
    block = f'<meta name="app-url" content="{base}">'
    charset = re.search(r"<meta\s+charset[^>]*>", html, re.I)
    if charset:
        pos = charset.end()
        return html[:pos] + "\n" + block + html[pos:]
    return html.replace("<head>", "<head>\n" + block, 1)


def rewrite_html(doc: Path, root: Path, base: str) -> int:
    text = doc.read_text(encoding="utf-8")
    text = set_meta(text, base)
    n = 0
    if base:
        def sub(m: re.Match) -> str:
            nonlocal n
            n += 1
            return base + "/" + canonical(doc, root, m.group(0))
        text = REL_RE.sub(sub, text)
    doc.write_bytes(text.encode("utf-8"))
    return n


def rewrite_json(doc: Path, base: str) -> int:
    try:
        data = json.loads(doc.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 0
    n = 0

    def fix_str(s: str) -> str:
        nonlocal n
        if EXACT_RE.fullmatch(s):
            n += 1
            return base + "/" + s if base else s
        if base:
            def md(m: re.Match) -> str:
                nonlocal n
                n += 1
                return m.group(1) + base + "/" + strip_dot(m.group(2)) + m.group(3)
            return MD_LINK_RE.sub(md, s)
        return s

    def walk(o):
        if isinstance(o, dict):
            return {k: walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [walk(v) for v in o]
        if isinstance(o, str):
            return fix_str(o)
        return o

    data = walk(data)
    if isinstance(data, dict):
        if isinstance(data.get("links"), dict):
            data["links"]["base"] = base
            data["links"]["mode"] = "absolute" if base else "internal"
        if isinstance(data.get("host"), dict):
            data["host"]["rawBase"] = base
            data["host"]["linkMode"] = "absolute" if base else "internal"
    if base or doc.name in ("catalog.json", "manifest.json"):
        doc.write_bytes((json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    return n


def apply(root: Path, base: str) -> None:
    html_n = json_n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or ".git" in p.parts:
            continue
        if p.suffix.lower() == ".html":
            html_n += rewrite_html(p, root, base)
        elif p.suffix.lower() == ".json":
            json_n += rewrite_json(p, base)
    print(f"prefixed {html_n} HTML links and {json_n} JSON paths with "
          f"APP_URL={base or '(empty — internal links kept)'}")


def link_or_copy(src: str, dst: str, follow_symlinks: bool = True) -> str:
    """Hard-link binaries (fast, no extra disk) but COPY text (edit safety)."""
    p = Path(src)
    if p.suffix.lower() in TEXT_EXT or p.name in (".env.example", ".gitattributes"):
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)
    try:
        os.link(src, dst)
        return dst
    except OSError:
        return shutil.copy2(src, dst, follow_symlinks=follow_symlinks)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--export", metavar="DIR", help="write an app-ready copy to DIR")
    g.add_argument("--in-place", action="store_true", help="rewrite the bundle itself")
    ap.add_argument("--bundle", help="path to the MaghzBook bundle (auto-detected)")
    args = ap.parse_args()

    bundle = find_bundle(args.bundle)
    base = app_url()
    if args.in_place:
        apply(bundle, base)
        print(f"updated {bundle}")
        return
    out = Path(args.export).resolve()
    if out == bundle:
        raise SystemExit("--export target must differ from the bundle; use --in-place")
    if out.exists():
        raise SystemExit(f"export target already exists: {out}")
    shutil.copytree(bundle, out, ignore=shutil.ignore_patterns(".git"),
                    copy_function=link_or_copy)
    apply(out, base)
    print(f"app-ready bundle written to {out}")


if __name__ == "__main__":
    main()
