"""Validate retained SR-image records without importing an ML runtime."""
import hashlib
from pathlib import Path


def validate_png(score, width, height, root):
    artifact = score['png']
    path = Path(artifact['path']).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('retained image escapes declared artifact root')
    if (artifact['width'], artifact['height']) != (width, height):
        raise ValueError('retained image dimensions differ')
    if artifact['format'] != 'PNG RGB8 sRGB lossless':
        raise ValueError('retained image format differs')
    digest = artifact['sha256']
    if len(digest) != 64 or path.name != digest + '.png':
        raise ValueError('invalid retained image identity')
    data = path.read_bytes()
    if len(data) != artifact['bytes'] or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError('retained image bytes changed')
