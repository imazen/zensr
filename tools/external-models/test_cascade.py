"""Cascade routing must preserve skipped tensors and final output pixels."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from cascade_contract import ARMS, validate


class CascadeTests(unittest.TestCase):
    def setUp(self):
        scratch=Path.home()/'tmp';scratch.mkdir(exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=scratch,prefix='zensr-cascade-test-')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.row=dict(q=50,width=8,height=4,low_width=2,low_height=1)
        def score(name,w,h):
            data=name.encode();digest=hashlib.sha256(data).hexdigest();p=self.root/(digest+'.png');p.write_bytes(data)
            return dict(ssim2='80',butter_max='1',butter_pnorm3='1',rgb8_sha256='a'*64,
                png=dict(path=str(p),sha256=digest,bytes=len(data),width=w,height=h,format='PNG RGB8 sRGB lossless'))
        self.cell=dict(row=self.row,plain_f32_sha256='b'*64,repair_f32_sha256='b'*64,
            routing=dict(applied=False,preview=score('preview',2,1)))
        self.cell['baseline']=score('baseline',8,4)
        for name in ARMS:self.cell[name]=score(name.split('_')[0],8,4)

    def test_skipped_outputs_are_identical(self):
        validate(self.cell,self.row,self.root)
        self.cell['dscf_repair']['ssim2']='79'
        with self.assertRaisesRegex(ValueError,'final output changed'):validate(self.cell,self.row,self.root)

    def test_skip_cannot_change_tensor(self):
        self.cell['repair_f32_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'input tensor changed'):validate(self.cell,self.row,self.root)

    def test_clean_cannot_be_repaired(self):
        self.row['q']=0;self.cell['routing']['applied']=True
        with self.assertRaisesRegex(ValueError,'clean input'):validate(self.cell,self.row,self.root)

    def test_restored_may_differ_but_must_be_finite(self):
        self.cell['routing']['applied']=True
        self.cell['dscf_repair']=copy.deepcopy(self.cell['dscf_repair'])
        self.cell['dscf_repair']['ssim2']='81'
        validate(self.cell,self.row,self.root)
        self.cell['spanf_repair']['butter_pnorm3']='NaN'
        with self.assertRaisesRegex(ValueError,'nonfinite'):validate(self.cell,self.row,self.root)
