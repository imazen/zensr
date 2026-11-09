"""Prevent partial or corrupt CPU timing evidence from becoming a final table."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from cpu_report import read_packet


class CPUReportTests(unittest.TestCase):
    def setUp(self):
        scratch=Path.home()/'tmp';scratch.mkdir(exist_ok=True)
        self.directory=tempfile.TemporaryDirectory(dir=scratch,prefix='zensr-cpu-report-test-')
        self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name)
        self.packet=dict(complete=True,placement=dict(status='enforced_uniform',enforcement='cpuset',
                core_class='performance',effective_cpus=[0,1],topology=dict(classes=[
                    dict(name='performance',cpus=[0,1],performance_rank=2),
                    dict(name='efficiency',cpus=[2,3],performance_rank=1)])),
            plan=dict(blocks=3,aa=False,baseline=dict(id='model'),candidates=[],
                cases=[dict(id='64-t1',fields=dict(n='64',threads='1'))],metric=dict(expected=3)),invocations=[])
        for block in range(3):
            output=self.root/str(block);output.mkdir()
            identity=dict(input_sha256='a'*64,output_sha256='b'*64,shape=[1,3,256,256])
            (output/'stdout.txt').write_text('IDENTITY '+json.dumps(identity)+'\n')
            (output/'stderr.txt').write_text('Maximum resident set size (kbytes): 1024\n')
            self.packet['invocations'].append(dict(case=0,arm=0,block=block,status='completed',exit_code=0,
                values_ns=[(block+1)*100_000_000]*3,output=str(output),correctness=['a'*64,[1,3,256,256]],
                artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}))

    def read(self):
        path=self.root/'packet.json.gz';path.write_bytes(gzip.compress(json.dumps(self.packet).encode()))
        return read_packet(path,'fixture')[0]

    def test_process_level_units_and_output_throughput(self):
        row=self.read()[0]
        self.assertEqual(row['median_process_mean_ms'],200)
        self.assertEqual(row['min_process_mean_ms'],100)
        self.assertEqual(row['max_process_mean_ms'],300)
        self.assertAlmostEqual(row['output_mpix_per_second'],.32768)

    def test_missing_process_is_rejected(self):
        self.packet['invocations'].pop()
        with self.assertRaisesRegex(ValueError,'incomplete or failed'):self.read()

    def test_corrupt_retained_stdout_is_rejected(self):
        (self.root/'0/stdout.txt').write_text('changed\n')
        with self.assertRaisesRegex(ValueError,'bytes changed'):self.read()

    def test_fresh_process_nondeterminism_is_rejected(self):
        row=self.packet['invocations'][0];path=self.root/'0/stdout.txt'
        data=json.loads(path.read_text()[len('IDENTITY '):]);data['output_sha256']='c'*64
        path.write_text('IDENTITY '+json.dumps(data)+'\n')
        row['artifacts'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,'hashes differ'):self.read()

    def test_missing_or_nonpositive_timing_is_rejected(self):
        self.packet['invocations'][0]['values_ns']=[0,1,2]
        with self.assertRaisesRegex(ValueError,'timing vector'):self.read()

    def test_placement_status_is_not_a_substring_search(self):
        self.packet['placement']['status']='uncontrolled'
        self.packet['placement']['diagnostic']='enforced_uniform was unavailable'
        with self.assertRaisesRegex(ValueError,'placement'):self.read()

    def test_mixed_core_and_efficiency_placement_are_rejected(self):
        for core_class,cpus in [('performance',[0,2]),('efficiency',[2,3])]:
            with self.subTest(core_class=core_class,cpus=cpus):
                self.packet['placement']['core_class']=core_class
                self.packet['placement']['effective_cpus']=cpus
                with self.assertRaisesRegex(ValueError,'placement'):self.read()


if __name__=='__main__':unittest.main()
