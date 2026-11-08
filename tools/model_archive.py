"""Shared model/archive packaging helpers; standard library only."""

import hashlib
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while b := stream.read(1024 * 1024):
            h.update(b)
    return h.hexdigest()


def make_archive(folder, destination, names):
    if destination.exists():
        raise ValueError("archive already exists")
    subprocess.run(
        [
            "tar",
            "-I",
            "zstd -T1 -3",
            "-cf",
            str(destination),
            "-C",
            str(folder),
            *names,
        ],
        check=True,
    )
    if destination.stat().st_size >= 2**31:
        raise ValueError("asset exceeds GitHub per-file limit")
    return dict(
        name=destination.name,
        bytes=destination.stat().st_size,
        sha256=digest(destination),
        members={n: digest(folder / n) for n in names},
    )


def verify_archive(path, members):
    """Verify every regular nested member against the exact source inventory."""
    process = subprocess.Popen(["zstd", "-dc", str(path)], stdout=subprocess.PIPE)
    found = set()
    try:
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                name = Path(member.name)
                if (
                    member.name not in members
                    or member.name in found
                    or not member.isfile()
                    or name.is_absolute()
                    or ".." in name.parts
                ):
                    raise ValueError("unexpected or unsafe archive member")
                found.add(member.name)
                digest = hashlib.sha256()
                with archive.extractfile(member) as source:
                    while block := source.read(1024 * 1024):
                        digest.update(block)
                if digest.hexdigest() != members[member.name]:
                    raise ValueError("archive member hash mismatch")
        if process.wait() != 0 or found != set(members):
            raise ValueError("incomplete archive")
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
