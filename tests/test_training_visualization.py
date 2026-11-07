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
TARGET=PROJECT/'notebooks/Visualization/training.ipynb'
BUILDER=PROJECT.parent/'meta/tools/build_training_visualization_notebook.py'


def source(cell):return ''.join(cell['source'])


class TrainingVisualizationTests(unittest.TestCase):
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

    def test_notebook_plain_schema_and_builder_freshness(self):
        nb=json.loads(TARGET.read_text(encoding='utf-8'));nbformat.validate(nbformat.from_dict(nb))
        spec=importlib.util.spec_from_file_location('training_viz_builder_test',BUILDER)
        b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b);b.build_notebook(check=True)
        for cell in nb['cells']:
            if cell['cell_type']!='code':continue
            compile(source(cell),cell['id'],'exec')
            self.assertEqual(cell['outputs'],[])
            for node in ast.walk(ast.parse(source(cell))):
                if isinstance(node,ast.ImportFrom):
                    self.assertEqual(node.level,0)
                    self.assertFalse((node.module or '').startswith(('phonetics','torch')))
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name):self.assertNotIn(node.func.id,{'exec','eval','display'})

    def test_builder_preserves_parameters(self):
        spec=importlib.util.spec_from_file_location('training_preserve_builder',BUILDER)
        b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training.ipynb'
            text='from pathlib import Path\nPRESENTATION_ID = "keep"\nTRAINING_ROOT = Path("/custom/logs")\n'
            path.write_text(json.dumps({'cells':[{'id':'parameters','source':text.splitlines(keepends=True)}]}))
            with patch.object(b,'TARGET',path):
                b.build_notebook();b.build_notebook(check=True)
            notebook=json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(source(next(c for c in notebook['cells'] if c['id']=='parameters')),text)

    def test_actual_standalone_prepare_reuse_and_no_weights(self):
        nb=json.loads(TARGET.read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);runs=make_training_fixture(root)
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            module=types.ModuleType('training_notebook_under_test');sys.modules[module.__name__]=module
            ns=module.__dict__
            try:
                for cell in nb['cells']:
                    if cell['cell_type']!='code':continue
                    if cell['id']=='parameters':
                        exec(source(cell),ns)
                        ns.update(TRAINING_RUNS=runs,OUTPUT_PROJECT_ROOT=root,SUMMARY_RUN_ID='summary',PRESENTATION_ID='report',MODE='prepare')
                        continue
                    with patch('numpy.load',side_effect=AssertionError('no weights or vectors')):
                        exec(source(cell),ns)
                report=ns['REPORT_PATH']
                for suffix in ('png','eps','metadata.json'):
                    self.assertEqual(len(list((report/'figures').glob('*.'+suffix))),5)
                manifest=json.loads((report/'manifest.json').read_text())
                self.assertEqual(manifest['status'],'complete')
                self.assertEqual(before,{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before})
                self.assertTrue((report/'tables/selected-checkpoints.md').exists())
                with self.assertRaises(FileExistsError):exec(source(nb['cells'][-1]),ns)
                ns.update(MODE='reuse',PRESENTATION_ID='rerender')
                def forbidden(*args,**kwargs):raise AssertionError('reuse must not read training logs')
                with patch.dict(ns,{'load_training_runs':forbidden}):exec(source(nb['cells'][-1]),ns)
                self.assertEqual(len(list((ns['REPORT_PATH']/'figures').glob('*.eps'))),5)
                self.assertEqual(ns['plt'].get_fignums(),[])
            finally:
                for name in ('_style_resources','_plot_cache'):
                    if name in ns:ns[name].cleanup()
                sys.modules.pop(module.__name__,None)


if __name__=='__main__':unittest.main()
