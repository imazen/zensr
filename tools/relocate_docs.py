#!/usr/bin/env python3
"""Move a tracked repository path and preserve relative Markdown links/references.

Invoke only from a claimed, clean checkout. Refuses to overwrite a destination.
Historical README prose retains original filenames; its hyperlinks still move.
"""
import argparse
from pathlib import Path
import posixpath
import re
from urllib.parse import unquote, urlsplit, urlunsplit
from check_docs import ROOT, LINK, files


def relocate(source, destination):
    source = Path(source)
    destination = Path(destination)
    for path in (source, destination):
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('paths must be inside the repository')
    if not (ROOT/source).exists() or (ROOT/destination).exists():
        raise ValueError('source must exist and destination must be new')
    names = files()
    moved = {name: destination/name.relative_to(source)
             for name in names if name == source or source in name.parents}
    if not moved:
        raise ValueError('no tracked/source files to relocate')
    mapping = {str(old): str(new) for old, new in moved.items()}
    references = re.compile(r'(?<![\w/.-])('+ '|'.join(
        re.escape(name) for name in sorted(mapping, key=len, reverse=True))+r')(?![\w.-])')
    rewritten = {}
    for old in names:
        if old.suffix not in ('.md', '.rs', '.py', '.txt', '.toml') and old.name != 'justfile':
            continue
        text = (ROOT/old).read_text()
        new = moved.get(old, old)
        links = []
        if old.suffix == '.md':
            def link(match):
                url = urlsplit(match.group(1))
                updated = match.group(0)
                if not (url.scheme or url.netloc or not url.path or url.path.startswith('/')):
                    target = posixpath.normpath(str(old.parent/ unquote(url.path)))
                    target = mapping.get(target, target)
                    relative = posixpath.relpath(target, str(new.parent))
                    href = urlunsplit(('', '', relative, url.query, url.fragment))
                    updated = match.group(0).replace(match.group(1), href)
                links.append(updated)
                return f'\x00LINK{len(links)-1}\x00'
            text = LINK.sub(link, text)
        if old.name != 'HISTORICAL_RESULTS.md':
            text = references.sub(lambda m: mapping[m.group(1)], text)
        for i, link_text in enumerate(links):
            text = text.replace(f'\x00LINK{i}\x00', link_text)
        if text != (ROOT/old).read_text():
            rewritten[new] = text
    (ROOT/destination).parent.mkdir(parents=True, exist_ok=True)
    (ROOT/source).rename(ROOT/destination)
    for name, text in rewritten.items():
        (ROOT/name).write_text(text)
    print(f'Moved {source} → {destination}; updated {len(rewritten)} text files.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source')
    p.add_argument('destination')
    args = p.parse_args()
    relocate(args.source, args.destination)
