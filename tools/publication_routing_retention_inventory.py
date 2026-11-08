#!/usr/bin/env python3
"""Check retained inputs and score metadata after the November routing cleanup.

This reads directory entries and score metadata, not pixel/map contents.
"""
import argparse
import collections
import hashlib
import json
import os
from pathlib import Path


def inventory(root):
    primary=json.loads((root/'development-primary/ARTIFACT_RETENTION.json').read_text())
    pilots=json.loads((root/'retention-plans-pilots-v1/SUMMARY.json').read_text())['runs']
    records={'development-primary':primary,**pilots}
    result={'runs':{},'removed_files':0,'removed_allocated_bytes':0}
    for name,record in records.items():
        assert record['stage']=='complete'
        counts=collections.Counter()
        for prefix in (root/name/'artifacts').iterdir():
            if not prefix.is_dir():continue
            with os.scandir(prefix) as entries:
                for entry in entries:
                    if not entry.is_file(follow_symlinks=False):
                        raise ValueError('nonregular retained artifact')
                    suffix='jpeg' if entry.name.endswith('.jpg') else ('rgb8' if entry.name.endswith('.rgb8.zst') else ('map' if entry.name.endswith('.butter-f32le.zst') else 'other'))
                    counts[suffix]+=1
        if sum(counts.values())!=record['protected_inputs'] or counts['map'] or counts['other']:
            raise ValueError('retained input inventory differs: '+name+' '+str(counts))
        for filename,expected in record['metadata'].items():
            h=hashlib.sha256()
            with (root/name/filename).open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            assert h.hexdigest()==expected,(name,filename)
        assert not (root/name/'AUDITED.json').exists()
        result['runs'][name]={'retained_inputs':dict(counts),'removed':record['removed_this_attempt'],'removed_allocated_bytes':record['unlinked_allocated_bytes_this_attempt'],'grid_coverage_verified':record.get('grid_coverage_verified',True),'score_metadata_unchanged':True}
        result['removed_files']+=record['removed_this_attempt']
        result['removed_allocated_bytes']+=record['unlinked_allocated_bytes_this_attempt']
    archive=root/'table-snapshots/primary-v1/cells.tar.zst'
    h=hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    assert h.hexdigest()=='0ff5a241849430ecad22b9bc911c94d15ed8d10d7844482b80d194d1b4fd4bfb'
    result['score_archive_sha256']=h.hexdigest()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    print(json.dumps(inventory(args.root), indent=2))
