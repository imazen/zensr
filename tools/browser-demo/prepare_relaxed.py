"""Build-only CNN arithmetic experiment from pinned, owned source; leaves native crates untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

SIMD_SHA='fa12285f44668d32830001bbfabfe1c0b225ac5eae7d4875f3c0de3129aa8ac7'

def sha(data):return hashlib.sha256(data).hexdigest()

def main(output, mode="relaxed"):
    source=Path(__file__).resolve().parent
    repo=source.parents[1]
    simd=(repo/'crates/zensr-micro/src/simd.rs').read_bytes()
    if sha(simd)!=SIMD_SHA:raise ValueError('CNN source changed; review the relaxed experiment before rebuilding')
    files={}
    for name in ['zensr-micro','zensr-zenjpeg']:
        crate=repo/'crates'/name
        for path in [crate/'Cargo.toml',*[path for folder in ['src','examples','benches'] for path in sorted((crate/folder).rglob('*'))]]:
            if path.is_file():files['crates/'+name+'/'+str(path.relative_to(crate))]=path.read_bytes()
    for path in [source/'repair/Cargo.toml',source/'repair/Cargo.lock',source/'repair/src/lib.rs']:
        files['repair/'+str(path.relative_to(source/'repair'))]=path.read_bytes()
    source_identity={name:sha(data) for name,data in files.items()}
    patched=simd.replace(b'.mul_add_portable(',b'.mul_add(').replace(b'.mul_sub_portable(',b'.mul_sub(')
    replacements=simd.count(b'.mul_add_portable(')+simd.count(b'.mul_sub_portable(')
    if replacements!=15:raise ValueError('Unexpected CNN arithmetic patch size')
    files['crates/zensr-micro/src/simd.rs']=f'// Generated {mode}-SIMD browser build; portable-rounding comments describe the strict reference.\n'.encode()+patched
    manifest=files['repair/Cargo.toml'].decode().replace('../../../crates/','../crates/')
    files['repair/Cargo.toml']=manifest.encode()
    output.mkdir(parents=True,exist_ok=False)
    for name,data in files.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    report=dict(mode=mode,source_simd_sha256=SIMD_SHA,replacements=replacements,
        policy=f'Only CNN vector multiply-add/sub changes from portable one-rounding to engine {mode} arithmetic; strict product source is unchanged.',
        source=source_identity,generated={name:sha(data) for name,data in files.items()})
    (output/'RELAXED_SOURCE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(str(output/'repair/Cargo.toml'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--mode',choices=['relaxed','unfused'],default='relaxed')
    args=parser.parse_args();main(args.output,args.mode)
