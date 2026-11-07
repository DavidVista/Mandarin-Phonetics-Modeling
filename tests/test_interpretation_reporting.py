"""Independent estimand, MC precision and Q–Q preparation checks."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from statistics import NormalDist
import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
from test_anchor_q_reporting import make_batch
from test_interpretation import build
from phonetics.evaluation.interpretation import MODELS
from phonetics.evaluation.interpretation_reporting import (
    build_q_group_tables,build_residual_qq_data,build_interpretation_presentation_data,
    write_interpretation_presentation_data,
)


def paired_fixture():
    batch = make_batch(models=MODELS)
    interpretation = build()
    for row in batch['results']:
        row.update(run_id='fixed',run_manifest_sha256='b'*64,checkpoint_sha256='c'*64)
    return batch,interpretation


class InterpretationReportingTests(unittest.TestCase):
    def test_twelve_nine_three_estimands(self):
        batch=make_batch(models=MODELS)
        for record in batch['results']:
            for row in record['score']['draw_scores']:
                row['score']=1. if row['edge_id'].endswith('->111') else .5
        result=build_q_group_tables(batch)
        self.assertEqual(len(result['summary']),60)
        self.assertEqual(len(result['seed_metrics']),300)
        self.assertEqual(len(result['seed_paired_deltas']),300)
        for row in result['summary']:
            expected={'all12':.625,'partial9':.5,'full3':1.}[row['group']]
            self.assertAlmostEqual(row['mean'],expected)
            count={'all12':12,'partial9':9,'full3':3}[row['group']]
            if row['metric']!='overall':count//=3
            self.assertEqual(row['transition_count'],count)
            self.assertAlmostEqual(row['transition_weight'],1/count)
            self.assertEqual(row['mc_se'],0.)

    def test_matched_seed_mc_and_paired_difference(self):
        profiles={(m,s):[0.,1.,0.,1.] for m in MODELS for s in range(5)}
        result=build_q_group_tables(make_batch(profiles,models=MODELS))
        for row in result['summary']:
            expected=np.std([0,1,0,1],ddof=1)/np.sqrt(4*row['transition_count'])
            self.assertAlmostEqual(row['mc_se'],expected)
            self.assertEqual(row['sample_sd'],0.)
        self.assertTrue(all(r['mc_se']==0 and r['mean']==0 for r in result['paired_deltas']))
        # Seeds can differ while their matched draw average is constant.
        profiles={}
        for model in MODELS:
            for seed,values in enumerate(([0,0,0,0],[1,1,1,1],[0,1,0,1],[1,0,1,0],[.5,.5,.5,.5])):
                profiles[(model,seed)]=values
        result=build_q_group_tables(make_batch(profiles,models=MODELS))
        self.assertTrue(all(r['mc_se']==0 and r['sample_sd']>0 for r in result['summary']))

    def test_single_draw_and_identity_mismatch(self):
        result=build_q_group_tables(make_batch(models=MODELS,draws=1))
        self.assertTrue(all(r['mc_se'] is None for r in result['summary']))
        batch=make_batch(models=MODELS)
        batch['results'][0]['score']['draw_scores'][0]['triplet_id']='wrong'
        with self.assertRaises(ValueError): build_q_group_tables(batch)

    def test_qq_each_seed_has_own_sample_sd_and_extremes(self):
        data={m:{str(s):[2.+s,3.+s,9.+s,12.+s] for s in range(5)} for m in MODELS}
        data['dimsim']=[1.,4.,6.,15.]
        result=build_residual_qq_data(data,max_quantiles=2)
        self.assertEqual(len(result['curves']),26)
        for row in result['curves']:
            y=np.array(data['dimsim'] if row['seed'] is None else data[row['model_id']][str(row['seed'])])
            np.testing.assert_allclose(row['standardized_residuals'],[(y.min()-y.mean())/y.std(ddof=1),(y.max()-y.mean())/y.std(ddof=1)])
            np.testing.assert_allclose(row['normal_quantiles'],[NormalDist().inv_cdf(.5/4),NormalDist().inv_cdf(3.5/4)])
        data['syllable']['0']=[1.,1.,1.,1.]
        row=build_residual_qq_data(data)['curves'][0]
        self.assertEqual(row['status'],'constant_residual')
        self.assertEqual(row['standardized_residuals'],[])

    def test_joint_tables_provenance_and_persistence(self):
        batch,interpretation=paired_fixture()
        prepared=build_interpretation_presentation_data(batch,interpretation)
        self.assertEqual(len(prepared['tables']),9)
        self.assertEqual(len(prepared['tables']['coefficients.md'].splitlines()),8)
        json.dumps(prepared,allow_nan=False)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);identity=root/'input.json';identity.write_text('{}')
            output=write_interpretation_presentation_data(root,'data',prepared,input_roots=[identity],source_hashes={'source':'d'*64})
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'complete')
            with self.assertRaises(FileExistsError):
                write_interpretation_presentation_data(root,'data',prepared,input_roots=[identity],source_hashes={'source':'d'*64})
        batch['results'][0]['checkpoint_sha256']='e'*64
        with self.assertRaises(ValueError): build_interpretation_presentation_data(batch,interpretation)


if __name__=='__main__':unittest.main()
