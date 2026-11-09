#!/usr/bin/env python3
"""Acquire pinned public SR research oracles without Git or shared-environment changes."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

REPOS = {
    's3diff': ('ArcticHare105/S3Diff', '6a2e0a47676f2f01fccdfd4e940077d3038d953e'),
    'resshift': ('zsyOAOA/ResShift', 'bb03b7d21614cace01787e097c8a6ab6b945227d'),
    'dscf': ('Yaozzz666/DSCF-SR', 'ef4a32d01b792120707d4cfd64f3993789a84d53'),
}


def model_metadata(root, model, revision):
    path = root/(model.replace('/', '-')+'-metadata.json')
    if path.exists():
        metadata = json.loads(path.read_text())
    else:
        url = f'https://huggingface.co/api/models/{model}/revision/{revision}?blobs=true'
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(2*1024*1024+1)
        if len(data) > 2*1024*1024:
            raise ValueError('model metadata exceeds bounded size')
        metadata = json.loads(data)
        if metadata['sha'] != revision:
            raise ValueError('HF revision differs')
        path.write_bytes(data)
    if metadata['sha'] != revision:
        raise ValueError('HF revision differs')
    return metadata


def fetch(url, path, limit, expected=None):
    if path.exists():
        h = hashlib.sha256()
        with path.open('rb') as existing:
            for block in iter(lambda: existing.read(1024*1024), b''):
                h.update(block)
        digest = h.hexdigest()
        if expected and digest != expected:
            raise ValueError(f'existing digest differs: {path}')
        return dict(url=url, bytes=path.stat().st_size, sha256=digest)
    part = path.with_suffix(path.suffix+'.partial')
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(url, timeout=120) as response, part.open('xb') as output:
        while block := response.read(1024*1024):
            size += len(block)
            if size > limit:
                raise ValueError(f'download exceeds bounded size: {url}')
            output.write(block)
            digest.update(block)
    if expected and digest.hexdigest() != expected:
        raise ValueError(f'download digest differs: {url}')
    part.rename(path)
    print(f'acquired {path.name}: {size} bytes', flush=True)
    return dict(url=url, bytes=size, sha256=digest.hexdigest())


def main(root, name):
    root.mkdir(parents=True, exist_ok=True)
    target = root/name
    target.mkdir(exist_ok=True)
    repo, revision = REPOS[name]
    archive = target/'source.tar.gz'
    records = dict(repository=repo, revision=revision, files={})
    records['source'] = fetch(f'https://codeload.github.com/{repo}/tar.gz/{revision}', archive, 256*1024*1024)
    source = target/'source'
    source.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar:
        for member in tar:
            parts = Path(member.name).parts[1:]
            if not parts or not member.isfile() or any(p in ('..', '') for p in parts):
                continue
            relative = Path(*parts)
            if relative.suffix not in ('.py', '.yaml', '.yml', '.json', '.md', '.txt') and relative.name not in ('LICENSE', 'LICENSE.txt'):
                continue
            path = source/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = tar.extractfile(member).read()
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError(f'extracted source changed: {path}')
            else:
                path.write_bytes(data)
    if name == 'dscf':
        filename = 'model_zoo/team23_DSCF.pth'
        path = target/'team23_DSCF.pth'
        records['files'][path.name] = fetch(
            f'https://raw.githubusercontent.com/{repo}/{revision}/{filename}', path, 535137,
            '239e76bc4e4e738491c3963805c0a79f524400acf5fbde34b2cb1a855dd62cb1')
    elif name == 'resshift':
        for filename, size in [('autoencoder_vq_f4.pth', 221364711), ('resshift_realsrx4_s4_v3.pth', 477917058), ('resshift_bicsrx4_s4.pth', 477917058)]:
            url = f'https://github.com/{repo}/releases/download/v2.0/{filename}'
            record = fetch(url, target/filename, size)
            if record['bytes'] != size:
                raise ValueError('release size differs')
            records['files'][filename] = record
    else:
        for model, revision, filenames in [
            ('zhangap/S3Diff', '41b5e9b6258829d379de50322fb843a97cacba27', ['s3diff.pkl','de_net.pth','README.md']),
            ('stabilityai/sd-turbo', 'b261bac6fd2cf515557d5d0707481eafa0485ec2', [
                'README.md','LICENSE.md','model_index.json','scheduler/scheduler_config.json',
                'text_encoder/config.json','text_encoder/model.safetensors',
                'unet/config.json','unet/diffusion_pytorch_model.safetensors',
                'vae/config.json','vae/diffusion_pytorch_model.safetensors',
                'tokenizer/merges.txt','tokenizer/special_tokens_map.json',
                'tokenizer/tokenizer_config.json','tokenizer/vocab.json'])]:
            metadata = model_metadata(root, model, revision)
            inventory = {v['rfilename']: v for v in metadata['siblings']}
            base = target/('sd-turbo' if model.startswith('stabilityai/') else 'weights')
            for filename in filenames:
                info = inventory[filename]
                path = base/filename
                path.parent.mkdir(parents=True, exist_ok=True)
                records['files'][str(path.relative_to(target))] = fetch(
                    f'https://huggingface.co/{model}/resolve/{revision}/{filename}', path,
                    info['size'], info.get('lfs', {}).get('sha256'))
    path = target/'ACQUISITION.json'
    if path.exists() and json.loads(path.read_text()) != records:
        raise ValueError('acquisition record changed')
    path.write_text(json.dumps(records, indent=2, sort_keys=True)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('model', choices=REPOS)
    a = p.parse_args()
    main(a.root, a.model)
