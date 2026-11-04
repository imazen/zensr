import tempfile
import csv
import hashlib
import json
from unittest.mock import patch
import publication_routing_analysis as study
from pathlib import Path
import unittest
import numpy as np
from sklearn.model_selection import GroupKFold
from publication_routing_analysis import (inputs, labels, origin_weights, summary,
    threshold_choices, choose, fit_model, predict, read_cells)


class PublicationIntegrity(unittest.TestCase):
    def test_size_representatives_ignore_outcomes_and_keep_singletons(self):
        from publication_routing_sizes import representatives
        rows=[]
        for i in range(9):
            row={name:str(i+1) for name in study.BASE+study.COEF+study.PIXELS}
            row.update(file=f'{i:03}',origin=f'origin-{i}',reference_sha256=f'ref-{i}',
                content='photo' if i<8 else 'singleton',encoder='turbo',ss='420',nominal_q='50',size='512',
                identity_ssim2='80',restored_ssim2='81')
            rows.append(row)
        before=representatives(rows,2)
        for row in rows:row['restored_ssim2']='0'
        self.assertEqual(before,representatives(rows,2))
        self.assertIn('ref-8',before[0])
        self.assertEqual(sum(c['origins'] for c in before[1]),9)

    def test_reference_columns_cannot_enter_inference(self):
        row=dict(width='32',probe_family='Mozjpeg',probe_scale='MozjpegQuality',ss='420',identity_ssim2='70')
        before=inputs([row],('width',))
        row['identity_ssim2']='99'
        np.testing.assert_array_equal(before,inputs([row],('width',)))
        with self.assertRaises(ValueError):inputs([row],('identity_ssim2',))
        with self.assertRaises(ValueError):inputs([row],('nominal_q',))
        with self.assertRaises(ValueError):inputs([row],('reference_metric',))

    def test_threshold_margins_match_both_runtime_scale_directions(self):
        gates=study.runtime_gates()
        rows=[dict(probe_scale='IjgQuality',probe_family='ImageMagick',probe_value='94',ss='420',estimated_gain='2'),
              dict(probe_scale='ButteraugliDistance',probe_family='Jpegli',probe_value='.6',ss='420',estimated_gain='2'),
              dict(probe_scale='MozjpegQuality',probe_family='Mozjpeg',probe_value='98',ss='420',estimated_gain='2')]
        choice=dict(skip=False,min_gain=.25,quality_margin=0.,distance_margin=0.)
        np.testing.assert_array_equal(study.threshold_decisions(rows,choice,gates),[True,True,True])
        choice.update(quality_margin=2.,distance_margin=.2)
        np.testing.assert_array_equal(study.threshold_decisions(rows,choice,gates),[False,False,False])
        choice.update(skip=True)
        np.testing.assert_array_equal(study.threshold_decisions(rows,choice,gates),[False,False,False])

    def test_exact_harm_curve_boundaries(self):
        rows=[dict(identity_ssim2='89.000',restored_ssim2='88.500'),
              dict(identity_ssim2='95.000',restored_ssim2='94.700'),
              dict(identity_ssim2='100.000',restored_ssim2='99.899')]
        gain,harm,limits=labels(rows)
        np.testing.assert_array_equal(harm,[False,False,True])
        np.testing.assert_array_equal(limits,[.5,.3,.1])

    def test_repeated_cells_do_not_multiply_origin_weight(self):
        w=origin_weights(np.array(['a','a','b']))
        self.assertEqual(w[:2].sum(),w[2])
        gain=np.array([2.,2.,-1.]);harm=np.array([False,False,True]);d=np.ones(3,bool)
        result=summary(gain,harm,np.full(3,.5),d,np.array(['a','a','b']))
        self.assertEqual(result['origin_mean'],.5)
        self.assertEqual(result['origin_harm'],.5)

    def test_folds_keep_all_variants_of_origin_together(self):
        groups=np.repeat(np.arange(25),8)
        folds=GroupKFold(n_splits=5,shuffle=True,random_state=7)
        visited=[]
        for train,valid in folds.split(np.zeros((200,1)),groups=groups):
            self.assertFalse(set(groups[train]) & set(groups[valid]))
            visited.extend(valid)
        self.assertEqual(sorted(visited),list(range(200)))

    def test_zero_budget_has_a_true_skip_all_candidate(self):
        gain=np.array([1.,-1.]);harm=np.array([False,True]);groups=np.array(['a','b'])
        candidates=threshold_choices(np.ones(2),np.zeros(2),gain,harm,groups)
        best=choose(candidates,0.)
        self.assertEqual(best['harm'],0.)
        self.assertEqual(best['coverage'],0.)

    def test_single_class_risk_fit_is_explicit_constant(self):
        x=np.array([[float(i),'Mozjpeg','MozjpegQuality','420'] for i in range(40)],dtype=object)
        model=fit_model(x,np.arange(40,dtype=float),np.zeros(40,bool),np.arange(40),('linear',1.),1)
        gain,risk=predict(model,x)
        self.assertTrue(np.isfinite(gain).all())
        np.testing.assert_array_equal(risk,np.zeros(40))

    def test_training_range_guard_rejects_extrapolation_and_unknown_categories(self):
        x=np.array([[float(i),'Mozjpeg','MozjpegQuality','420'] for i in range(40)],dtype=object)
        model=fit_model(x,np.arange(40,dtype=float),np.zeros(40,bool),np.arange(40),('linear',1.),1)
        query=x[:4].copy();query[0,0]=-1.;query[1,0]=39.;query[2,1]='unseen-family'
        domain=study.in_domain(model,query)
        np.testing.assert_array_equal(domain,[False,True,False,True])
        choice=dict(skip=False,penalty=0.,threshold=0.,domain_guard=True)
        np.testing.assert_array_equal(study.decisions(np.ones(4),np.zeros(4),choice,domain),domain)
        with self.assertRaisesRegex(ValueError,'missing fitted-domain'):
            study.decisions(np.ones(4),np.zeros(4),choice)

    def test_selection_freeze_and_reserved_evaluation_roundtrip(self):
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='synthetic-roundtrip-',dir=base))
        def dataset(name,role,count):
            folder=root/name;(folder/'cells').mkdir(parents=True)
            for i in range(count):
                feature=(i%11)/10
                row=dict(file=f'{name}-{i}.png',content='synthetic',origin=f'{name}-{i//2}',
                    role=role,ref_kind='synthetic',reference_sha256=hashlib.sha256(f'{name}-{i}'.encode()).hexdigest(),
                    width='32',height='32',size='32',encoder='fixture',ss='420',nominal_q='50',
                    probe_family='ImageMagick',probe_scale='IjgQuality',probe_value='50',
                    estimated_gain='.5',bpp='1',gate='false',conservative_gate='false',
                    identity_ssim2='80',restored_ssim2=str(79+2*feature),coef_zero_ac_blocks=str(feature))
                for metric in ('psnr','butter_max','butter_pnorm3'):
                    row['identity_'+metric]='1';row['restored_'+metric]='1'
                with (folder/'cells'/f'{i:04}.tsv').open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row),delimiter='\t');writer.writeheader();writer.writerow(row)
            digest=hashlib.sha256()
            for path in sorted((folder/'cells').glob('*.tsv')):
                digest.update(path.name.encode());digest.update(path.read_bytes())
            (folder/'AUDITED.json').write_text(json.dumps({'cells_sha256':digest.hexdigest()}))
            (folder/'COMPLETE').write_text('synthetic test fixture\n')
            (folder/'IDENTITY.tsv').write_text(f'role\t{role}\nbinary_sha256\tsynthetic\n')
            (folder/'FEATURES.tsv').write_text('synthetic feature identities\n')
            (folder/'ENCODERS.tsv').write_text('synthetic-encoder\tsynthetic-sha\n')
            return folder
        development=dataset('dev','router_development',80)
        reserved=dataset('test','locked_test',20)
        with patch.object(study,'FEATURES',{'coefficient':study.BASE+('coef_zero_ac_blocks',)}), patch.object(study,'CONFIGS',(('linear',1.),)):
            calls=0
            def interrupted_predict(model,x):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('synthetic interruption')
                return predict(model,x)
            with patch.object(study,'predict',interrupted_predict):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    study.fit(development,root/'selection')
            checkpoint=root/'selection'/'coefficient-linear-1.fold-0.npz'
            checkpoint_stat=checkpoint.stat()
            study.fit(development,root/'selection')
            self.assertEqual(checkpoint.stat().st_mtime_ns,checkpoint_stat.st_mtime_ns)
            with self.assertRaisesRegex(ValueError,'already frozen'):
                study.fit(development,root/'selection')
        self.assertTrue((root/'selection'/'selection.freeze.json').is_file())
        study.evaluate(reserved,root/'selection',root/'report.json')
        report=json.loads((root/'report.json').read_text())
        self.assertEqual(report['policies']['unchanged']['origin_mean'],0.)
        self.assertEqual(report['policies']['oracle_positive']['harm_count'],0)
        self.assertIn('paired_vs_fidelity',report['policies']['coefficient/budget-0'])
        (reserved/'IDENTITY.tsv').write_text('role\tlocked_test\nbinary_sha256\tsynthetic\nsizes\t64\nmanifest_sha256\tsynthetic-subset\n')
        with self.assertRaisesRegex(ValueError,'frozen study contract'):
            study.evaluate(reserved,root/'selection',root/'wrong-primary-report.json')
        study.evaluate(reserved,root/'selection',root/'size-report.json',root/'report.json')
        self.assertEqual(json.loads((root/'size-report.json').read_text())['evaluation_scope'],'size_generalization')
        (reserved/'IDENTITY.tsv').write_text('role\tlocked_test\nbinary_sha256\tsynthetic\nencoders\t'+','.join(study.MODE_ENCODERS)+'\nmanifest_sha256\tsynthetic-subset\nencoder_dependencies_sha256\tsynthetic-subset\nexclusions\t\n')
        study.evaluate(reserved,root/'selection',root/'mode-report.json',mode_report=root/'report.json')
        self.assertEqual(json.loads((root/'mode-report.json').read_text())['evaluation_scope'],'mode_generalization')
        (reserved/'ENCODERS.tsv').write_text('synthetic-encoder\tchanged-binary\n')
        with self.assertRaisesRegex(ValueError,'encoder binaries'):
            study.evaluate(reserved,root/'selection',root/'changed-mode-report.json',mode_report=root/'report.json')
        (reserved/'FEATURES.tsv').write_text('changed feature identities\n')
        with self.assertRaisesRegex(ValueError,'frozen study contract'):
            study.evaluate(reserved,root/'selection',root/'changed-report.json')

    def test_missing_complete_marker_fails(self):
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            with self.assertRaises(ValueError):read_cells(Path(directory),'router_development')


if __name__=='__main__':unittest.main()
