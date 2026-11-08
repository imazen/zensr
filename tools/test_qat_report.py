import unittest
from decimal import Decimal
from qat import report

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

    def test_missing_duplicate_or_changed_control_is_rejected(self):
        rows=self.rows()
        with self.assertRaisesRegex(ValueError,'coverage'):report.summarize(rows[:-1],'forced')
        rows[-1]['default_restore']='0'
        with self.assertRaisesRegex(ValueError,'control'):report.summarize(rows,'forced')

if __name__=='__main__':unittest.main()
