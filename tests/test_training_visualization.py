"""Standalone training notebook, artist semantics and actual PNG/EPS exports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import nbformat

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'tests'),str(PROJECT/'src')]
from test_training_summary import make_training_fixture
from phonetics.evaluation.training_summary import load_training_runs
from phonetics.visualization import load_protocol,apply_style
from phonetics.visualization.training import build_training_figures


def source(cell):return ''.join(cell['source'])


class TrainingVisualizationTests(unittest.TestCase):
    def test_curves_keep_distinct_epoch_budgets_on_shared_axis(self):
        with tempfile.TemporaryDirectory() as folder:
            runs = make_training_fixture(Path(folder), character=True, epochs_by_model={'syllable': 2})
            figures = build_training_figures(load_training_runs(runs))
            try:
                syllable, components = figures[0].axes[:2]
                self.assertEqual(list(syllable.lines[0].get_xdata()), [1, 2])
                self.assertEqual(list(components.lines[0].get_xdata()), [1, 2, 3])
                self.assertEqual(syllable.get_xlim(), (1, 3))
                self.assertEqual(components.get_xlim(), (1, 3))
            finally:
                for figure in figures:
                    plt.close(figure)

    def test_panel_semantics_legends_bands_and_centering(self):
        with tempfile.TemporaryDirectory() as folder:
            data=load_training_runs(make_training_fixture(Path(folder)))
            p=load_protocol();apply_style(p);figs=build_training_figures(data,p)
            try:
                self.assertEqual([len(f.axes) for f in figs],[5,4,2,3,1])
                self.assertIn('Selected epoch',[t.get_text() for t in figs[0].legends[0].texts])
                for ax in figs[0].axes:
                    self.assertTrue(all(v==2 for v in ax.lines[-1].get_xdata()))
                    self.assertEqual(ax.lines[-1].get_markerfacecolor(),'white')
                self.assertIn('± SD across seeds',[t.get_text() for t in figs[2].legends[0].texts])
                self.assertEqual([ax.get_title() for ax in figs[3].axes],['Components','Combined λ1','Combined λ10'])
                for ax in figs[3].axes:self.assertEqual(len(ax.lines),3)
                center=lambda ax:ax.get_position().x0+ax.get_position().width/2
                self.assertAlmostEqual(center(figs[3].axes[2]),(center(figs[3].axes[0])+center(figs[3].axes[1]))/2)
                self.assertAlmostEqual(np.mean([center(a) for a in figs[0].axes[3:]]),np.mean([center(a) for a in figs[0].axes[:3]]))
                np.testing.assert_allclose(figs[4].axes[0].lines[0].get_ydata(),np.array(data['models']['components']['curves']['validation']['invalid_if_rate']['mean'])*100)
                self.assertAlmostEqual(figs[4].get_size_inches()[0]*25.4,90.)
                for f in figs:
                    for ax in f.axes:
                        self.assertTrue(ax.collections)
                        for collection in ax.collections:
                            self.assertTrue(np.all(collection.get_facecolors()[:,3]==1.))
            finally:
                for f in figs:plt.close(f)






    def test_sdk_prepare_reuse_and_no_weights(self):
        from phonetics.visualization.training import run_training_visualization
        with tempfile.TemporaryDirectory(dir=PROJECT) as folder:
            root=Path(folder); runs=make_training_fixture(root)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            kwargs=dict(training_runs=runs,summary_run_id='summary',source_hashes={'test':'a'*64},source_digest='b'*64)
            with patch('numpy.load',side_effect=AssertionError('no weights or vectors')):
                report=run_training_visualization(root,'report',mode='prepare',**kwargs)
            self.assertEqual(len(list((report/'figures').glob('*.png'))),5)
            self.assertEqual(len(list((report/'figures').glob('*.eps'))),5)
            self.assertTrue((report/'tables/selected-checkpoints.md').exists())
            self.assertEqual(before,{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before})
            with self.assertRaises(FileExistsError):run_training_visualization(root,'report',mode='prepare',**kwargs)
            with patch('phonetics.visualization.training.load_training_runs',side_effect=AssertionError('no reanalysis')):
                second=run_training_visualization(root,'rerender',mode='reuse',**kwargs)
            self.assertEqual(len(list((second/'figures').glob('*.png'))),5)
            self.assertEqual(plt.get_fignums(),[])

if __name__=='__main__':unittest.main()
