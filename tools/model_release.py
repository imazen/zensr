#!/usr/bin/env python3
"""Capture model binaries with ID-based catalog/provenance for draft assets."""
import argparse
import collections
import json
from pathlib import Path
import shutil

from pair_release import digest, json_file, make_archive


def portable_metadata(meta):
    result = {k: meta[k] for k in ['arch','scale','nf','nc','space','total_floats','f16_goldens'] if k in meta}
    source = meta.get('source', '')
    if source and not source.startswith('/') and '/home/' not in source and '/Users/' not in source:
        result['source'] = source
    repro = meta.get('repro', {})
    result['training'] = {k: repro[k] for k in ['seed_torch','seed_numpy','commit','torch','device','amp_dtype','init_sha256_16'] if k in repro}
    result['training']['trainer'] = Path(repro.get('trainer','unknown')).name
    data = repro.get('dataset_meta') or {}
    result['training']['dataset'] = {k: data[k] for k in ['n','val_tail','seed','crop','q','encoders','gen2','gen3','val_split'] if k in data}
    return result


def build(inputs, output):
    output.mkdir(parents=True, exist_ok=False)
    payload = output / 'payload'
    payload.mkdir()
    objects = payload / 'objects'
    objects.mkdir()
    catalog = []
    omitted = []
    files = []
    for host in sorted(inputs.iterdir()):
        if not host.is_dir():continue
        for path in sorted(host.rglob('*')):
            if not path.is_file():continue
            rel = path.relative_to(host)
            if path.suffix in ['.json','.sh']:
                omitted.append(dict(capture=host.name,path=str(rel),bytes=path.stat().st_size,
                                    sha256=digest(path),reason='original metadata or local-path command; portable metadata supplied'))
                continue
            sha = digest(path)
            stored = objects / (sha + path.suffix)
            if not stored.exists():shutil.copyfile(path,stored)
            files.append(dict(capture=host.name,path=str(rel),bytes=path.stat().st_size,
                              sha256=sha,object=stored.relative_to(payload).as_posix()))
        adopted = host/'models/adopted'
        if adopted.exists():
            for model in sorted(adopted.iterdir()):
                if not model.is_dir():continue
                meta_path=model/'meta.json'
                meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
                members=[f for f in files if f['capture']==host.name and Path(f['path']).parent==Path('models/adopted')/model.name]
                # Catalog records model names and binary basenames, not capture paths.
                info=dict(name=model.name,metadata=portable_metadata(meta),
                          evaluation_status='historical_not_revalidated',
                          original_metadata_sha256=digest(meta_path) if meta_path.exists() else None,
                          files=[{**{k:f[k] for k in ['bytes','sha256','object']},'name':Path(f['path']).name} for f in members])
                existing=next((c for c in catalog if c['name']==info['name']),None)
                if existing:
                    if {f['sha256'] for f in existing['files']} != {f['sha256'] for f in info['files']}:
                        existing.setdefault('additional_captures',[]).append(info)
                else:catalog.append(info)
    known={m['name'] for m in catalog}
    extra=[]
    for f in files:
        rel=Path(f['path'])
        if str(rel).startswith('models/adopted/'):continue
        checkpoint='checkpoints' in rel.parts
        associated=next((n for n in sorted(known,key=len,reverse=True) if rel.name.startswith(n+'_')),None)
        kind = 'routing-estimator' if rel.suffix == '.joblib' else 'training-checkpoint' if checkpoint else 'runtime-baseline'
        extra.append(dict(kind=kind,
                          model=associated,object=f['object'],sha256=f['sha256'],bytes=f['bytes'],
                          name=None if checkpoint else rel.name))
    public=dict(schema=1,models=catalog,other_binaries=extra,
        notes=['Historical research artifacts, not a fresh model ranking.',
               'Older quality-model lineage includes the documented corrupted-target history.',
               'Original local-path commands are not portable retraining recipes.',
               'See per-model metadata; unknown provenance or licence is not upgraded by archiving.',
               'Checkpoint payloads use PyTorch serialization; load weights_only=True.'])
    json_file(payload/'MODEL-CATALOG.json',public)
    (payload/'README.md').write_text('''# Model artifact preview\n\nMODEL-CATALOG.json maps model names and original binary basenames to\ncontent-addressed objects. Copy the selected objects to their catalogued names\nto use an adopted-model directory. Metadata records architecture, scale and\nrecoverable training provenance. Objects are byte-preserved; repeated captures\nshare an object. Original local-path metadata/commands are retained outside\nthis asset and are inventoried separately.\n\nThe collection includes historical models, runtime baselines and training\ncheckpoints. It is not a production recommendation or a new quality comparison.\nSome historical model lineage used corrupted targets or superseded data splits.\nThird-party adopted weights retain their upstream terms; consult the source\nresearch record before redistribution beyond this unpublished preview.\n\nhttps://github.com/imazen/zensr/blob/main/docs/MODELS.md\n''')
    names=['MODEL-CATALOG.json','README.md',*[p.relative_to(payload).as_posix() for p in sorted(objects.iterdir())]]
    # make_archive supports nested model object paths; the pair extractor remains flat-only.
    asset=make_archive(payload,output/'zensr-model-collection-20261107.tar.zst',names)
    json_file(output/'model-asset.json',asset)
    json_file(output/'not-uploaded-original-metadata.json',omitted)
    summary=dict(adopted_model_names=len(catalog),captured_binary_files=len(files),
                 unique_binary_objects=len(list(objects.iterdir())),
                 checkpoints=sum(f['kind']=='training-checkpoint' for f in extra),
                 routing_estimators=sum(f['kind']=='routing-estimator' for f in extra),
                 runtime_baselines=sum(f['kind']=='runtime-baseline' for f in extra),
                 original_text_files_not_uploaded=len(omitted),asset_bytes=asset['bytes'],asset_sha256=asset['sha256'])
    json_file(output/'summary.json',summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.inputs.resolve(),args.output.resolve())
