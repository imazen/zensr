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
    def test_concurrent_staging_transfers_publish_identical_bytes(self):
        from concurrent.futures import ThreadPoolExecutor
        from publication_routing_stage import transfer
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='staging-concurrent-',dir=base))
        data=b'identical artifact'*1000;name=hashlib.sha256(data).hexdigest()+'.zst'
        destination=root/'archive'/name;sources=[]
        for i in range(4):
            source=root/str(i)/name;source.parent.mkdir();source.write_bytes(data);sources.append(source)
        with ThreadPoolExecutor(max_workers=4) as pool:
            sizes=list(pool.map(lambda source:transfer(source,destination),sources))
        self.assertEqual(sizes,[len(data)]*4)
        self.assertEqual(destination.read_bytes(),data)
        self.assertTrue(all(not source.exists() for source in sources))

    def test_staging_guard_does_not_resume_after_transfer_failure(self):
        import publication_routing_stage as staging
        from types import SimpleNamespace
        capacity=[SimpleNamespace(f_bavail=gib*1024**3,f_frsize=1) for gib in (24,24,24,15,15)]
        with patch.object(staging.os,'statvfs',side_effect=capacity), \
             patch.object(staging.subprocess,'check_output',return_value='active\n'), \
             patch.object(staging.subprocess,'run') as control, \
             patch.object(staging,'drain',side_effect=OSError('archive unavailable')):
            with self.assertRaisesRegex(OSError,'archive unavailable'):
                staging.watch(Path('source'),Path('destination'),'writer.scope')
        control.assert_called_once_with(['systemctl','--user','freeze','writer.scope'],check=True)

    def test_staging_transfer_preserves_conflicts_and_requires_durability(self):
        import publication_routing_stage as staging
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='staging-transfer-',dir=base))
        data=b'preserve these artifact bytes';name=hashlib.sha256(data).hexdigest()+'.rgb8.zst'
        source=root/'source'/name;source.parent.mkdir();source.write_bytes(data)
        destination=root/'destination'/name
        with patch.object(staging,'sync_directory',side_effect=OSError('sync failed')):
            with self.assertRaisesRegex(OSError,'sync failed'):staging.transfer(source,destination)
        self.assertEqual(source.read_bytes(),data)
        self.assertEqual(destination.read_bytes(),data)
        self.assertEqual(staging.transfer(source,destination),len(data))
        self.assertFalse(source.exists());self.assertEqual(destination.read_bytes(),data)
        source.write_bytes(data);destination.write_bytes(b'conflicting bytes')
        with self.assertRaisesRegex(ValueError,'conflicting destination'):staging.transfer(source,destination)
        self.assertEqual(source.read_bytes(),data)
        self.assertEqual(destination.read_bytes(),b'conflicting bytes')
        source.write_bytes(b'incomplete')
        with self.assertRaisesRegex(ValueError,'corrupt staged'):staging.transfer(source,destination)
        self.assertEqual(source.read_bytes(),b'incomplete')

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

    def test_native_supplement_retains_large_origins_and_eligible_files(self):
        from publication_routing_sizes import representatives,native_representatives
        rows=[];references=[]
        for i in range(8):
            row={name:str(i+1) for name in study.BASE+study.COEF+study.PIXELS}
            row.update(file=f'{i:03}',origin=f'origin-{i//2}',reference_sha256=f'ref-{i}',
                content='photo',encoder='turbo',ss='420',nominal_q='50',size='512',
                identity_ssim2='80',restored_ssim2='81')
            rows.append(row)
            references.append(dict(reference_sha256=f'ref-{i}',width=4096 if i%2 else 1024,height=512))
        base,_=representatives(rows,2)
        selected,clusters=native_representatives(rows,references,2)
        self.assertTrue(set(base)<=set(selected))
        self.assertTrue({'ref-1','ref-3','ref-5','ref-7'}<=set(selected))
        self.assertEqual(len(selected),len(set(selected)))
        self.assertEqual({c['members'][0] for c in clusters if c['native_floor']==4096},
                         {f'origin-{i}' for i in range(4)})
        for row in rows:row['restored_ssim2']='-100'
        self.assertEqual((selected,clusters),native_representatives(rows,references,2))

    def test_resume_quarantines_only_unreferenced_corruption(self):
        from publication_routing_resume import verify
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            root=Path(folder);(root/'cells').mkdir();(root/'artifacts'/'aa').mkdir(parents=True)
            valid=root/'artifacts'/'aa'/(hashlib.sha256(b'valid').hexdigest()+'.jpg');valid.write_bytes(b'valid')
            bad=root/'artifacts'/'aa'/('0'*64+'.jpg');bad.write_bytes(b'interrupted')
            columns=('jpeg_artifact','reference_artifact','decoded_artifact','restored_artifact','identity_butter_map','restored_butter_map')
            with (root/'cells'/'cell.tsv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=columns,delimiter='\t');writer.writeheader()
                writer.writerow({key:str(valid.relative_to(root)) for key in columns})
            with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):verify(root)
            self.assertTrue(bad.exists())
            record=verify(root,True)
            self.assertFalse(bad.exists());self.assertEqual(valid.read_bytes(),b'valid')
            self.assertEqual((root/record['quarantined'][0]).read_bytes(),b'interrupted')
            valid.write_bytes(b'corrupt referenced bytes');bad.write_bytes(b'another interrupted file')
            with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):verify(root,True)
            self.assertEqual(valid.read_bytes(),b'corrupt referenced bytes')
            self.assertTrue(bad.exists())

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

    def test_numeric_scaling_uses_origin_weights(self):
        x=np.array([[0.,'a'],[0.,'a'],[0.,'a'],[10.,'b']],dtype=object)
        groups=np.array(['first','first','first','second'])
        model=fit_model(x,np.array([0.,0.,0.,1.]),np.zeros(4,dtype=bool),groups,('linear',1.),1)
        scaler=model[0].named_transformers_['numeric']
        self.assertAlmostEqual(scaler.mean_[0],5.)
        self.assertAlmostEqual(scaler.var_[0],25.)

    def test_weighted_constant_scaling_is_finite_without_warnings(self):
        import warnings
        values=np.column_stack((np.full(460,.5),np.full(460,50.),np.arange(460,dtype=float)+1e6))
        weights=origin_weights(np.array(['dense']*400+[str(i) for i in range(60)]))
        with warnings.catch_warnings():
            warnings.simplefilter('error',RuntimeWarning)
            scaler=study.OriginScaler().fit(values,sample_weight=weights)
        np.testing.assert_array_equal(scaler.var_[:2],[0.,0.])
        np.testing.assert_array_equal(scaler.transform(values)[:,:2],np.zeros((460,2)))
        mean=np.average(values[:,2],weights=weights)
        variance=np.average((values[:,2]-mean)**2,weights=weights)
        self.assertAlmostEqual(scaler.mean_[2],mean)
        self.assertAlmostEqual(scaler.var_[2],variance)

    def test_harmed_origin_interval_counts_origins_and_handles_zero_events(self):
        groups=np.repeat(np.arange(20),3);decision=np.ones(60,bool);harm=np.zeros(60,bool)
        zero=study.harmed_origin_interval(harm,decision,groups)
        self.assertEqual(zero['interval_95'][0],0.)
        self.assertAlmostEqual(zero['interval_95'][1],1-.025**(1/20))
        harm[:3]=True
        repeated=study.harmed_origin_interval(harm,decision,groups)
        self.assertEqual(repeated['harmed_origins'],1)
        self.assertEqual(repeated['fraction'],.05)
        harm[:]=True
        self.assertEqual(study.harmed_origin_interval(harm,decision,groups)['interval_95'][1],1.)
        decision[:]=False
        self.assertEqual(study.harmed_origin_interval(harm,decision,groups),zero)

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

    def test_shallow_tree_gain_risk_and_serialization(self):
        import joblib
        values=np.linspace(-3.,3.,2000)
        x=np.array([[v,'Mozjpeg','MozjpegQuality','420'] for v in values],dtype=object)
        gain=np.where(np.abs(values)<1.,2.,-1.);harm=gain<-.5
        train=np.arange(0,len(x),2);valid=np.arange(1,len(x),2)
        model=fit_model(x[train],gain[train],harm[train],train,('hist',3),1)
        pg,pr=predict(model,x[valid])
        self.assertGreater(np.mean((pg>0)==(gain[valid]>0)),.95)
        self.assertGreater(np.mean((pr>.5)==harm[valid]),.95)
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as folder:
            path=Path(folder)/'model.joblib';joblib.dump(model,path)
            replay=predict(joblib.load(path),x[valid])
            np.testing.assert_array_equal(pg,replay[0]);np.testing.assert_array_equal(pr,replay[1])

    def test_selection_freeze_and_reserved_evaluation_roundtrip(self):
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        root=Path(tempfile.mkdtemp(prefix='synthetic-roundtrip-',dir=base))
        def dataset(name,role,count,sizes=(512,),source_name=None):
            folder=root/name;(folder/'cells').mkdir(parents=True)
            coverage=[]
            source_name=source_name or name
            for i,size in ((i,size) for i in range(count) for size in sizes):
                feature=(i%11)/10
                row=dict(file=f'{source_name}-{i}.png',content='synthetic',origin=f'{source_name}-{i//2}',
                    role=role,ref_kind='synthetic',reference_sha256=hashlib.sha256(f'{source_name}-{i}'.encode()).hexdigest(),
                    width=str(size),height=str(size),size=str(size),encoder='fixture',ss='420',nominal_q='50',
                    probe_family='ImageMagick',probe_scale='IjgQuality',probe_value='50',
                    estimated_gain='.5',bpp='1',gate='false',conservative_gate='false',
                    identity_ssim2='80',restored_ssim2=str(79+2*feature),coef_zero_ac_blocks=str(feature))
                for artifact in ('jpeg_artifact','reference_artifact','decoded_artifact','restored_artifact','identity_butter_map','restored_butter_map'):
                    row[artifact]='synthetic-'+artifact
                for arm in ('identity','restored'):
                    for channel in ('r','g','b'):row[f'{arm}_sse_{channel}']='1'
                for metric in ('psnr','butter_max','butter_pnorm3'):
                    row['identity_'+metric]='1';row['restored_'+metric]='1'
                with (folder/'cells'/f'{i:04}-{size}.tsv').open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row),delimiter='\t');writer.writeheader();writer.writerow(row)
                coverage.append(dict(file=row['file'],size=row['size'],encoder=row['encoder'],ss=row['ss'],q=row['nominal_q'],status='required'))
            with (folder/'COVERAGE.tsv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(coverage[0]),delimiter='\t');writer.writeheader();writer.writerows(coverage)
            digest=hashlib.sha256()
            for path in sorted((folder/'cells').glob('*.tsv')):
                digest.update(path.name.encode());digest.update(path.read_bytes())
            (folder/'AUDITED.json').write_text(json.dumps({'cells':count*len(sizes),'cells_sha256':digest.hexdigest(),
                'audit_source_sha256':study.sha(Path(study.__file__).with_name('publication_routing_cells.py')),
                'coverage_sha256':study.sha(folder/'COVERAGE.tsv')}))
            (folder/'COMPLETE').write_text('synthetic test fixture\n')
            (folder/'IDENTITY.tsv').write_text(f'role\t{role}\nbinary_sha256\tsynthetic\nsizes\t{",".join(map(str,sizes))}\nmanifest_sha256\tsynthetic-{"subset" if len(sizes)>1 else "full"}\n')
            (folder/'FEATURES.tsv').write_text('synthetic feature identities\n')
            (folder/'ENCODERS.tsv').write_text('synthetic-encoder\tsynthetic-sha\n')
            return folder
        development=dataset('dev','router_development',80)
        reserved=dataset('test','locked_test',20)
        ledger=reserved/'COVERAGE.tsv';original_ledger=ledger.read_bytes()
        ledger.write_bytes(original_ledger+b'\n')
        with self.assertRaisesRegex(ValueError,'coverage changed after artifact audit'):study.read_cells(reserved,'locked_test')
        ledger.write_bytes(original_ledger)
        from publication_routing_tables import pack
        snapshot=root/'dev-snapshot';pack(development,snapshot)
        self.assertEqual(study.read_cells(snapshot,'router_development'),study.read_cells(development,'router_development'))
        archive=snapshot/'cells.tar.zst';archive_bytes=archive.read_bytes();archive.write_bytes(archive_bytes+b'changed')
        with self.assertRaisesRegex(ValueError,'snapshot archive changed'):study.read_cells(snapshot,'router_development')
        archive.write_bytes(archive_bytes)
        marker=snapshot/'COVERAGE.tsv';original_marker=marker.read_bytes();marker.write_bytes(original_marker+b'\n')
        with self.assertRaisesRegex(ValueError,'snapshot metadata changed'):study.read_cells(snapshot,'router_development')
        marker.write_bytes(original_marker)
        from publication_routing_tables import pack
        (development/'selection.json').write_text('{"fixture": "selection provenance"}\n')
        selected_snapshot=root/'snapshot-with-selection'
        pack(development,selected_snapshot)
        self.assertEqual((selected_snapshot/'selection.json').read_bytes(),(development/'selection.json').read_bytes())
        self.assertEqual(study.read_cells(selected_snapshot,'router_development'),study.read_cells(development,'router_development'))
        (selected_snapshot/'selection.json').write_text('changed')
        with self.assertRaisesRegex(ValueError,'snapshot metadata changed'):
            study.read_cells(selected_snapshot,'router_development')
        (snapshot/'selection.json').write_text('unbound')
        with self.assertRaisesRegex(ValueError,'unbound snapshot selection'):
            study.read_cells(snapshot,'router_development')
        dense=dataset('dev-sizes','router_development',20,study.STUDY_SIZES,'dev')
        primary_rows,primary_digest=study.read_cells(development,'router_development')
        dense_rows,dense_digest=study.read_cells(dense,'router_development')
        dense_selection=dict(input_cells_sha256=primary_digest,selected_manifest_sha256='synthetic-subset',
            source_manifest_sha256='synthetic-full',features=list(study.BASE+study.COEF+study.PIXELS),
            selector_sha256=study.sha(Path(study.__file__).with_name('publication_routing_sizes.py')),
            sizes=list(study.STUDY_SIZES),clusters_per_content=2,seed=20261104,
            native_rule=dict(cluster_floor=2048,all_origins_floor=4096),
            clusters=[dict(representative_sha256=sha) for sha in sorted({r['reference_sha256'] for r in dense_rows})])
        (dense/'selection.json').write_text(json.dumps(dense_selection))
        combined,_,provenance=study.development_cells(development,dense)
        self.assertEqual(len(combined),460)
        self.assertEqual(provenance['identical_replays'],20)
        outside=[dict(r,origin='outside') for r in dense_rows]
        with patch.object(study,'read_cells',side_effect=[(primary_rows,primary_digest),(outside,dense_digest)]):
            with self.assertRaisesRegex(ValueError,'outside primary development'):study.development_cells(development,dense)
        replay=[dict(r,restored_ssim2='0') if r['size']=='512' else r for r in dense_rows]
        with patch.object(study,'read_cells',side_effect=[(primary_rows,primary_digest),(replay,dense_digest)]):
            with self.assertRaisesRegex(ValueError,'replayed development cell differs'):study.development_cells(development,dense)

        with patch.object(study,'FEATURES',{'coefficient':study.BASE+('coef_zero_ac_blocks',)}), patch.object(study,'CONFIGS',(('linear',1.),)):
            calls=0
            def interrupted_predict(model,x):
                nonlocal calls
                calls+=1
                if calls==2:raise RuntimeError('synthetic interruption')
                return predict(model,x)
            with patch.object(study,'predict',interrupted_predict):
                with self.assertRaisesRegex(RuntimeError,'synthetic interruption'):
                    study.fit(development,root/'selection',dense)
            checkpoint=root/'selection'/'coefficient-linear-1.fold-0.npz'
            checkpoint_stat=checkpoint.stat()
            study.fit(development,root/'selection',dense)
            self.assertEqual(checkpoint.stat().st_mtime_ns,checkpoint_stat.st_mtime_ns)
            with self.assertRaisesRegex(ValueError,'already frozen'):
                study.fit(development,root/'selection',dense)
        self.assertTrue((root/'selection'/'selection.freeze.json').is_file())
        real_sha=study.sha
        with patch.object(study,'sha',side_effect=lambda p:'changed-selector' if Path(p).name=='publication_routing_sizes.py' else real_sha(p)):
            with self.assertRaisesRegex(ValueError,'selector changed after fitting'):
                study.evaluate(reserved,root/'selection',root/'changed-selector-report.json')
        study.evaluate(reserved,root/'selection',root/'report.json')
        report=json.loads((root/'report.json').read_text())
        self.assertEqual(report['policies']['unchanged']['origin_mean'],0.)
        self.assertEqual(report['policies']['oracle_positive']['harm_count'],0)
        self.assertEqual(report['policies']['oracle_positive']['worst_origin_cases'],[])
        cases=report['policies']['ungated']['worst_origin_cases']
        self.assertEqual(cases[0]['gain'],-1.)
        self.assertEqual(len(cases),len({c['origin'] for c in cases}))
        self.assertEqual([c['gain'] for c in cases],sorted(c['gain'] for c in cases))
        self.assertTrue(all(c['jpeg_artifact']=='synthetic-jpeg_artifact' for c in cases))
        self.assertIn('paired_vs_fidelity',report['policies']['coefficient/budget-0'])
        for policy in report['policies'].values():
            point=policy['operating_point']
            if point and 'model_config' in point:
                expected=None if point['selection']['skip'] else (root/'selection'/f"{point['selection']['model']}.joblib").stat().st_size
                self.assertEqual(point['router_artifact_bytes'],expected)
        (reserved/'IDENTITY.tsv').write_text('role\tlocked_test\nbinary_sha256\tsynthetic\nsizes\t64\nmanifest_sha256\tsynthetic-subset\n')
        with self.assertRaisesRegex(ValueError,'frozen study contract'):
            study.evaluate(reserved,root/'selection',root/'wrong-primary-report.json')
        with self.assertRaisesRegex(ValueError,'reserved size inventory'):
            study.evaluate(reserved,root/'selection',root/'wrong-size-report.json',root/'report.json')
        reserved_sizes=dataset('test-sizes','locked_test',10,study.STUDY_SIZES,'test')
        size_rows,_=study.read_cells(reserved_sizes,'locked_test')
        reserved_selection=dict(dense_selection,input_cells_sha256=report['test_cells_sha256'],
            clusters=[dict(representative_sha256=h) for h in sorted({r['reference_sha256'] for r in size_rows})])
        (reserved_sizes/'selection.json').write_text(json.dumps(reserved_selection))
        study.evaluate(reserved_sizes,root/'selection',root/'size-report.json',root/'report.json')
        self.assertEqual(json.loads((root/'size-report.json').read_text())['evaluation_scope'],'size_generalization')
        reserved_selection['input_cells_sha256']='different-primary'
        (reserved_sizes/'selection.json').write_text(json.dumps(reserved_selection))
        with self.assertRaisesRegex(ValueError,'reserved representative selection'):
            study.evaluate(reserved_sizes,root/'selection',root/'wrong-selection-report.json',root/'report.json')
        (reserved/'IDENTITY.tsv').write_text('role\tlocked_test\nbinary_sha256\tsynthetic\nsizes\t512\nencoders\t'+','.join(study.MODE_ENCODERS)+'\nmanifest_sha256\tsynthetic-subset\nencoder_dependencies_sha256\tsynthetic-subset\nexclusions\t\n')
        with self.assertRaises(FileNotFoundError):
            study.evaluate(reserved,root/'selection',root/'missing-mode-selection.json',mode_report=root/'report.json')
        reserved_selection['input_cells_sha256']=report['test_cells_sha256']
        reserved_selection['clusters']=[dict(representative_sha256=h) for h in report['test_reference_sha256s']]
        mode_selection=json.dumps(reserved_selection)
        (reserved/'selection.json').write_text(json.dumps(reserved_selection))
        study.evaluate(reserved,root/'selection',root/'mode-report.json',mode_report=root/'report.json')
        reserved_selection['clusters']=[]
        (reserved/'selection.json').write_text(json.dumps(reserved_selection))
        with self.assertRaisesRegex(ValueError,'representative membership'):
            study.evaluate(reserved,root/'selection',root/'wrong-mode-members.json',mode_report=root/'report.json')
        (reserved/'selection.json').write_text(mode_selection)
        self.assertEqual(json.loads((root/'mode-report.json').read_text())['evaluation_scope'],'mode_generalization')
        (reserved/'ENCODERS.tsv').write_text('synthetic-encoder\tchanged-binary\n')
        with self.assertRaisesRegex(ValueError,'encoder binaries'):
            study.evaluate(reserved,root/'selection',root/'changed-mode-report.json',mode_report=root/'report.json')
        (reserved/'FEATURES.tsv').write_text('changed feature identities\n')
        with self.assertRaisesRegex(ValueError,'frozen study contract'):
            study.evaluate(reserved,root/'selection',root/'changed-report.json')

    def test_psnr_exact_match_transitions_do_not_use_99_sentinel(self):
        rows=[]
        for before,after in ((4,1),(0,1),(1,0),(0,0)):
            row={'identity_psnr':'99','restored_psnr':'99'}
            for arm,value in (('identity',before),('restored',after)):
                for c in ('r','g','b'):row[f'{arm}_sse_{c}']=str(value)
            rows.append(row)
        groups=np.asarray(['a','b','c','d'])
        result=study.psnr_changes(rows,np.ones(4,dtype=bool),groups)
        self.assertIsNone(result['origin_mean_improvement'])
        self.assertEqual(result['negative_infinite_change_count'],1)
        self.assertEqual(result['positive_infinite_change_count'],1)
        self.assertEqual(result['exact_to_exact_count'],1)
        self.assertAlmostEqual(result['origin_weighted_mean_finite_change'],5*np.log10(4))
        self.assertEqual(result['origin_fraction_worse'],.25)
        self.assertEqual(result['worst_kind'],'negative_infinity')
        unchanged=study.psnr_changes(rows,np.zeros(4,dtype=bool),groups)
        self.assertEqual(unchanged['origin_mean_improvement'],0.)
        self.assertEqual(unchanged['origin_fraction_worse'],0.)

    def test_missing_complete_marker_fails(self):
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            with self.assertRaises(ValueError):read_cells(Path(directory),'router_development')

    def test_deduplicated_artifact_still_checks_each_rows_dimensions(self):
        import publication_routing_cells as cells
        base=Path.home()/'tmp'/'zensr-publication-tests';base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as directory:
            root=Path(directory);(root/'cells').mkdir();coverage=[]
            for i in range(2):
                row=dict(file=str(i),reference_sha256=str(i),size='32',encoder='fixture',ss='420',nominal_q='50',
                         width='8' if i==0 else '16',height='16' if i==0 else '8',jpeg_artifact='fake.jpg',
                         reference_artifact='fake.rgb8.zst',decoded_artifact='fake.rgb8.zst',restored_artifact='fake.rgb8.zst',
                         identity_butter_map='fake.butter-f32le.zst',restored_butter_map='fake.butter-f32le.zst')
                with (root/'cells'/f'{i}.tsv').open('w') as f:
                    writer=csv.DictWriter(f,fieldnames=list(row),delimiter='\t');writer.writeheader();writer.writerow(row)
                coverage.append(dict(file=str(i),size='32',encoder='fixture',ss='420',q='50',status='required'))
            with (root/'COVERAGE.tsv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(coverage[0]),delimiter='\t');writer.writeheader();writer.writerows(coverage)
            (root/'COMPLETE').write_text('synthetic fixture\n')
            def inspect(path):
                return dict(dimensions=(8,16),ss='420') if path.suffix=='.jpg' else dict(bytes=512 if 'butter' in path.name else 384)
            with patch.object(cells,'inspect_artifact',side_effect=inspect) as inspection:
                with self.assertRaisesRegex(ValueError,'decoder dimensions differ'):cells.audit(root,workers=4)
                self.assertEqual(inspection.call_count,3)
            self.assertFalse((root/'AUDITED.json').exists())
            with patch.object(cells,'inspect_artifact',return_value=dict(dimensions=(8,16),ss='444')):
                with self.assertRaisesRegex(ValueError,'subsampling differs'):cells.audit(root,workers=4)
            self.assertFalse((root/'AUDITED.json').exists())
            with patch.object(cells,'inspect_artifact',side_effect=ValueError('artifact hash mismatch')):
                with self.assertRaisesRegex(ValueError,'artifact hash mismatch'):cells.audit(root,workers=4)
            self.assertFalse((root/'AUDITED.json').exists())
            failed=json.loads((root/'AUDIT_PROGRESS.json').read_text())
            self.assertEqual(failed['stage'],'failed')
            self.assertIn('artifact hash mismatch',failed['error'])
            self.assertEqual(failed['required_cells'],2)
            # A subsequent successful audit must replace the failure record
            # only after every row and the final coverage gate pass.
            row['width']='8';row['height']='16'
            with (root/'cells'/'1.tsv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=list(row),delimiter='\t');writer.writeheader();writer.writerow(row)
            binary=root/'fixture-zstd';binary.write_bytes(b'fixture')
            with patch.object(cells,'inspect_artifact',side_effect=inspect),patch.object(cells,'zstd_program',return_value=str(binary)):
                cells.audit(root,workers=4)
            complete=json.loads((root/'AUDIT_PROGRESS.json').read_text())
            result=json.loads((root/'AUDITED.json').read_text())
            self.assertEqual(complete['stage'],'complete')
            self.assertIsNone(complete['error'])
            self.assertEqual(complete['checked_cells'],2)
            self.assertEqual(result['cells'],2)
            self.assertEqual(result['artifacts'],3)
            self.assertEqual(result['coverage_sha256'],cells.digest(root/'COVERAGE.tsv'))
            self.assertEqual(list(root.glob('.*.partial')),[])


if __name__=='__main__':unittest.main()
