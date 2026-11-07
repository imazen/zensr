#!/usr/bin/env python3
"""Check repository-relative Markdown link paths (not external URLs or anchors)."""
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'\[[^\]]*\]\(([^)]+)\)')


def files():
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=ROOT).decode().split('\0')
    return sorted({Path(name) for name in names if name and (ROOT/name).is_file()})


def main():
    failures = []
    checked = 0
    for name in files():
        if name.suffix != '.md':
            continue
        for match in LINK.finditer((ROOT/name).read_text()):
            target = urlsplit(match.group(1))
            if target.scheme or target.netloc or not target.path or target.path.startswith('/'):
                continue
            checked += 1
            if not (ROOT/name.parent/unquote(target.path)).exists():
                failures.append(f'{name}: {match.group(1)}')
    if failures:
        raise SystemExit('Broken documentation links:\n'+'\n'.join(failures))
    print(f'{checked} repository-relative documentation links resolve.')


if __name__ == '__main__':
    main()
