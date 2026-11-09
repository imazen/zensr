"""Assemble ignored browser assets from hash-identified retained artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import numpy_helper


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(a):
    source = Path(__file__).resolve().parent
    dist = source / 'dist'
    dist.mkdir(exist_ok=True)
    assets = ['index.html', 'style.css', 'app.mjs', 'worker.mjs', 'tile-worker.mjs',
              'repair-worker.mjs', 'decode-worker.mjs', 'runtime.mjs', 'pool.mjs',
              'pool-plan.mjs', 'core.mjs', 'budgets.mjs', 'viewer.mjs', 'gallery.mjs']
    for name in assets:
        shutil.copy2(source / name, dist / name)
    vendor = dist / 'vendor'
    vendor.mkdir(exist_ok=True)
    runtime = source / 'node_modules/onnxruntime-web'
    for path in (runtime / 'dist').iterdir():
        if path.suffix == '.wasm' or path.name.startswith('ort-wasm-') and path.suffix == '.mjs' or path.name in ['ort.webgpu.bundle.min.mjs', 'ort.wasm.bundle.min.mjs']:
            shutil.copy2(path, vendor / path.name)
    shutil.copy2(source / 'ORT-LICENSE', vendor / 'ORT-LICENSE')
    for name, folder, stem in [('codec', 'codec', 'zensr_browser_codec'), ('jpeg', 'jpeg-probe', 'zensr_browser_jpeg_probe')]:
        target = dist / name
        target.mkdir(exist_ok=True)
        generated = source / folder / 'pkg'
        for suffix in ['.js', '_bg.wasm']:
            shutil.copy2(generated / (stem + suffix), target / (stem + suffix))
    repair_target=dist/'repair'
    repair_target.mkdir(exist_ok=True)
    for suffix in ['.js','_bg.wasm']:
        shutil.copy2(source/'repair/pkg'/('zensr_browser_repair'+suffix),repair_target/('zensr_browser_repair'+suffix))
    models = dist / 'models'
    models.mkdir(exist_ok=True)
    manifest = {}
    for name, path in [('spanf', a.sr / 'models/SPANF_x4_256x256.onnx'), ('dscf', a.dscf / 'models/DSCF_x4_256x256.onnx')]:
        graph = onnx.load(path)
        original_ops = [node.SerializeToString() for node in graph.graph.node]
        del graph.graph.value_info[:]
        for value in [*graph.graph.input, *graph.graph.output]:
            for i, label in [(2, 'height'), (3, 'width')]:
                dim = value.type.tensor_type.shape.dim[i]
                dim.ClearField('dim_value')
                dim.dim_param = ('output_' if value in graph.graph.output else '') + label
        if name == 'spanf':
            original = {t.name: numpy_helper.to_array(t) for t in graph.graph.initializer}
            order = ['conv_near.weight']
            for b in range(1, 6):
                for c in ['c1_r', 'c2_r', 'c3_r']:
                    order.extend([f'block_{b}.{c}.eval_conv.weight', f'block_{b}.{c}.eval_conv.bias'])
            order += ['conv_cat.weight','conv_cat.bias','conv_2.eval_conv.weight','conv_2.eval_conv.bias']
            if b''.join(original[k].astype('<f4').tobytes() for k in order) != (a.sr/'models/spanf_weights.raw').read_bytes():
                raise ValueError('SPANF initializer identity differs')
            mixed = bytearray()
            for tensor in graph.graph.initializer:
                if tensor.name in order and tensor.name.endswith('.weight'):
                    tensor.CopyFrom(numpy_helper.from_array(original[tensor.name].astype('<f2').astype('<f4'), tensor.name))
            for key in order:
                mixed += original[key].astype('<f2' if key.endswith('.weight') else '<f4').tobytes()
            if mixed != (a.sr/'models/spanf_weights_f16.raw').read_bytes():
                raise ValueError('SPANF stored-precision identity differs')
        if original_ops != [node.SerializeToString() for node in graph.graph.node]:
            raise ValueError('model operations changed')
        onnx.checker.check_model(graph)
        target = models / (name + '.onnx')
        onnx.save(graph, target)
        manifest[name] = dict(url='./models/' + target.name, sha256=sha(target),
                              source_sha256=sha(path), precision='FP32 compute', halo=16 if name=='spanf' else 21)
    weights=source.parents[1]/'models/adopted/dejpeg_rt24i_canon4/weights_f16.raw'
    shutil.copy2(weights,models/'canon4.raw')
    manifest['repair']=dict(url='./models/canon4.raw',sha256=sha(weights),precision='canon4 stored f16 weights, FP32 compute', relaxed=bool(a.relaxed_package))
    (dist/'models.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if a.relaxed_package:
        target=dist/'repair-relaxed'
        target.mkdir(exist_ok=True)
        for name in ['zensr_browser_repair.js','zensr_browser_repair_bg.wasm']:
            shutil.copy2(a.relaxed_package/name,target/name)
    print(json.dumps(dict(directory=str(dist),models=manifest),indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sr',type=Path,required=True)
    p.add_argument('--dscf',type=Path,required=True)
    p.add_argument('--relaxed-package',type=Path)
    main(p.parse_args())
