"""Synthetic training-log verification, selection, weighting and persistence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from phonetics.evaluation.training_summary import (
    TRAINING_MODELS,METRICS_SOURCE_SHA256,SELECTION_RULE,load_training_runs,
    write_training_summary,load_training_summary,
)
from phonetics.research.artifacts import file_sha256


def write_json(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False),encoding='utf-8')


def make_training_fixture(root):
    runs={}
    for model in TRAINING_MODELS:
        run=root/'runs'/model/'exact-run';artifacts={};results=[]
        def save(name,value):
            path=run/name;write_json(path,value);artifacts[name]=file_sha256(path)
        for seed in range(5):
            history=[]
            for epoch in range(1,4):
                row={'epoch':epoch}
                for split in ('train','validation'):
                    base=([1.,.4,.4][epoch-1] if split=='validation' else [1.2,.6,.5][epoch-1])+.01*seed
                    values={'characters':200 if split=='validation' else 1000,'steps':2}
                    if model=='syllable':
                        values.update(syllable_ce=base,objective=base,syllable_accuracy=.4+.15*epoch+.001*seed)
                    if model=='components' or model.startswith('components-acoustic'):
                        count=5-epoch+seed
                        values.update(initials_ce=base*1.2,finals_ce=base,tones_ce=base*.8,component_ce=base,
                            objective=base,initials_accuracy=.5+.1*epoch,finals_accuracy=.6+.08*epoch,
                            tones_accuracy=.7+.06*epoch,joint_ift_accuracy=.4+.12*epoch,
                            invalid_if_count=count,invalid_if_rate=count/values['characters'])
                    if model=='acoustic' or model.startswith('components-acoustic'):
                        weight=10. if model.endswith('lambda-10') else 1.
                        values.update(cosine=base*.1,weighted_cosine=weight*base*.1)
                        values['objective']=values.get('component_ce',0.)+values['weighted_cosine']
                    row[split]=values
                history.append(row)
            best=history[1]
            test=copy.deepcopy(best['validation']);test['characters']=300
            if 'invalid_if_count' in test:test['invalid_if_rate']=test['invalid_if_count']/300
            summary={'seed':seed,'status':'complete','best_epoch':2,'best_validation_objective':best['validation']['objective'],
                     'best_validation_metrics':best['validation'],'test_metrics':test}
            save(f'seed-{seed}/history.json',{'epochs':history});save(f'seed-{seed}/summary.json',summary)
            results.append(summary)
        save('summary.json',{'status':'complete','seeds_completed':5,'seed_results':results})
        # These files deliberately do not exist. The loader must never open weights/events.
        artifacts.update({'seed-0/best.pt':'f'*64,'seed-0/events.jsonl':'e'*64})
        manifest={'schema_version':1,'run_type':'training_run','status':'complete','run_id':run.name,
            'model_id':model,'seeds':list(range(5)),'seed_status':{str(s):'complete' for s in range(5)},
            'selection':SELECTION_RULE,'code_sha256':{'metrics.py':METRICS_SOURCE_SHA256},'artifacts':artifacts,
            'experiment_id':'synthetic','experiment_hashes':{'manifest.json':'a'*64,'splits.json':'b'*64},
            'config':{'training':{'epochs':3},'training_seeds':list(range(5))},
            'lambda_cosine':10. if model.endswith('lambda-10') else 1. if model.endswith('lambda-1') else None}
        write_json(run/'manifest.json',manifest)
        runs[model]={'path':run,'manifest_sha256':file_sha256(run/'manifest.json')}
    return runs


def rehash(runs,model,relative):
    run=runs[model]['path'];manifest=json.loads((run/'manifest.json').read_text())
    manifest['artifacts'][relative]=file_sha256(run/relative)
    write_json(run/'manifest.json',manifest);runs[model]['manifest_sha256']=file_sha256(run/'manifest.json')


class TrainingSummaryTests(unittest.TestCase):
    def test_realistic_selection_uses_validation_objective_earliest_tie(self):
        with tempfile.TemporaryDirectory() as folder:
            runs=make_training_fixture(Path(folder));data=load_training_runs(runs)
            self.assertEqual(len(data['selected']),25)
            self.assertEqual(data['diagnostics'],[])
            self.assertTrue(all(r['best_epoch']==2 for r in data['selected']))
            # Epoch3 accuracy is higher, but objective ties and earliest epoch wins.
            self.assertLess(data['selected'][0]['validation']['syllable_accuracy'],.85)
            curve=data['models']['syllable']['curves']['validation']['objective']
            np.testing.assert_allclose(curve['sample_sd'],np.std(np.arange(5)*.01,ddof=1))
            self.assertAlmostEqual(curve['mean'][0],1.02)

    def test_hash_identity_epoch_and_split_failures(self):
        for mode in ('hash','seed','epoch','split','selection','metric_code'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as folder:
                runs=make_training_fixture(Path(folder));root=runs['components']['path']
                if mode=='hash':(root/'seed-0/history.json').write_text('{}')
                elif mode in ('epoch','split'):
                    path=root/'seed-0/history.json';h=json.loads(path.read_text())
                    if mode=='epoch':h['epochs'][1]['epoch']=8
                    else:h['epochs'][0]['train']['characters']=999
                    write_json(path,h);rehash(runs,'components','seed-0/history.json')
                else:
                    path=root/'manifest.json';m=json.loads(path.read_text())
                    if mode=='seed':m['seeds']=[0,1,2,3]
                    elif mode=='selection':m['selection']='maximum accuracy'
                    else:m['code_sha256']['metrics.py']='0'*64
                    write_json(path,m);runs['components']['manifest_sha256']=file_sha256(path)
                with self.assertRaises((ValueError,KeyError)):load_training_runs(runs)

    def test_definition_mismatch_and_missing_metric(self):
        with tempfile.TemporaryDirectory() as folder:
            runs=make_training_fixture(Path(folder));root=runs['components']['path']
            path=root/'seed-0/history.json';h=json.loads(path.read_text())
            h['epochs'][0]['train']['component_ce']*=3
            write_json(path,h);rehash(runs,'components','seed-0/history.json')
            with self.assertRaisesRegex(ValueError,'CE definition'):load_training_runs(runs)
        with tempfile.TemporaryDirectory() as folder:
            runs=make_training_fixture(Path(folder));root=runs['components']['path']
            path=root/'seed-0/history.json';h=json.loads(path.read_text())
            del h['epochs'][0]['validation']['initials_accuracy']
            write_json(path,h);rehash(runs,'components','seed-0/history.json')
            data=load_training_runs(runs)
            self.assertNotIn('initials_accuracy',data['models']['components']['curves']['validation'])
            self.assertTrue(any(r.get('metric')=='initials_accuracy' for r in data['diagnostics']))

    def test_summary_roundtrip_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runs=make_training_fixture(root);data=load_training_runs(runs)
            path=write_training_summary(root,'summary',data,source_hashes={'source':'c'*64})
            loaded,_=load_training_summary(path)
            self.assertEqual(loaded,data)
            self.assertIn('Mean | Seed SD',(path/'tables/selected-summary.md').read_text())
            with self.assertRaises(FileExistsError):write_training_summary(root,'summary',data,source_hashes={'source':'c'*64})


if __name__=='__main__':unittest.main()
