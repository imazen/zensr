"""Coverage and unchanged-path gates for the four-arm SR cascade experiment."""
import math
from sr_artifacts import validate_png

ARMS = ('spanf_direct', 'spanf_repair', 'dscf_direct', 'dscf_repair')


def validate(cell, row, artifacts):
    if cell['row'] != row:
        raise ValueError('cascade input identity differs')
    applied = cell['routing']['applied']
    if type(applied) is not bool:
        raise ValueError('invalid repair decision')
    if row['q'] == 0 and applied:
        raise ValueError('clean input was repaired')
    for name in ('plain_f32_sha256', 'repair_f32_sha256'):
        if len(cell[name]) != 64:
            raise ValueError('missing input tensor identity')
    if not applied and cell['plain_f32_sha256'] != cell['repair_f32_sha256']:
        raise ValueError('routed-out input tensor changed')
    preview = cell['routing']['preview']
    validate_png(preview, row['low_width'], row['low_height'], artifacts)
    for arm in ('baseline',) + ARMS:
        score = cell[arm]
        for key in ('ssim2', 'butter_max', 'butter_pnorm3'):
            if not math.isfinite(float(score[key])):
                raise ValueError('nonfinite cascade score')
        if len(score['rgb8_sha256']) != 64:
            raise ValueError('missing output pixel identity')
        validate_png(score, row['width'], row['height'], artifacts)
    if not applied:
        for model in ('spanf', 'dscf'):
            if cell[model + '_direct'] != cell[model + '_repair']:
                raise ValueError('routed-out final output changed')
