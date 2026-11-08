import unittest
from decimal import Decimal
from qat import report, merge

class ReportTests(unittest.TestCase):
    def test_decimal_harm_boundary(self):
        self.assertEqual(report.limit(Decimal('89.9')),Decimal('.5'))
        self.assertEqual(report.limit(Decimal('95')),Decimal('.30'))
        self.assertEqual(report.limit(Decimal('100')),Decimal('.10'))
        self.assertFalse(Decimal('99.9')-Decimal('100') < -report.limit(Decimal('100')))

    def rows(self):
        values=[]
        for arm in ['baseline','ptq8']+[f'{kind}-s{seed}-selected' for seed in [7,17,29] for kind in ['float','qat8']]:
            for i,origin in enumerate(['a','a','a','b']):
                score=Decimal('80')+((Decimal('1') if origin=='a' else Decimal('-1')) if arm!='baseline' else 0)
                values.append(dict(id=str(i),source_id=origin,origin=origin,arm=arm,
                    identity_ssim2='70',forced_ssim2=str(score),routed_ssim2=str(score),
                    default_restore='1',encoder='zenjpeg',ss='420',q='25',
                    forced_butter_pnorm3='1',identity_butter_pnorm3='2'))
        return values

    def test_origin_balance_and_complete_paired_seed_reporting(self):
        result=report.summarize(self.rows(),'routed')
        self.assertEqual(result['arms']['ptq8']['mean_vs_baseline_ssim2'],0)
        self.assertEqual(len(result['paired_comparisons']),6)
        self.assertEqual(result['qat_seed_range_vs_baseline'],[0,0])
        self.assertEqual(result['qat_across_seeds']['baseline']['mean'],0)
        self.assertEqual(result['qat_across_seeds']['matched_float']['exploratory_seed_origin_bootstrap_95'],[0,0])

    def test_postquantized_float_controls_remain_paired(self):
        rows=self.rows()
        rows += [dict(r,arm=r['arm'].replace('float-s','float-ptq-s').replace('-selected',''))
                 for r in rows.copy() if r['arm'].startswith('float-s')]
        result=report.summarize(rows,'routed')
        self.assertEqual(result['qat_across_seeds']['matched_float_ptq']['mean'],0)
        self.assertEqual(result['paired_comparisons']['qat8-s7-vs-float-ptq-s7']['mean'],0)

    def test_missing_duplicate_or_changed_control_is_rejected(self):
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'coverage'):report.summarize(rows[:-1],'forced')
        rows[-1]['default_restore']='0'
        with self.assertRaisesRegex(ValueError,'control'):report.summarize(rows,'forced')

class MergeTests(unittest.TestCase):
    def fixtures(self):
        base=dict(id='one',arm='baseline',origin='a',source_id='1007',identity_ssim2='70',
                  default_restore='1',forced_ssim2='80',routed_ssim2='80')
        extra=dict(base,arm='ptq8-gpu')
        return [base],[base.copy(),extra]

    def test_exact_repeated_baseline_is_required(self):
        first,second=self.fixtures()
        self.assertEqual(len(merge.combine(first,second)),1)
        second[0]['forced_ssim2']='80.00000000000001'
        with self.assertRaisesRegex(ValueError,'baseline differs'):merge.combine(first,second)

    def test_extra_control_cannot_change_routing(self):
        first,second=self.fixtures();second[1]['default_restore']='0'
        with self.assertRaisesRegex(ValueError,'control differs'):merge.combine(first,second)

    def test_three_postquantized_controls_have_complete_baseline_coverage(self):
        first,second=self.fixtures()
        controls=[second[0]]+[dict(second[1],arm=f'float-ptq-s{s}') for s in [7,17,29]]
        self.assertEqual(len(merge.combine(first,controls)),3)
        with self.assertRaisesRegex(ValueError,'coverage'):merge.combine(first,controls[:-1]+[controls[1]])

if __name__=='__main__':unittest.main()
