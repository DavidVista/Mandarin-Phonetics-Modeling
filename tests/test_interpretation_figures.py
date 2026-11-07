"""Artist assertions and real opaque PNG/EPS export of new interpretation views."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
from test_interpretation_reporting import paired_fixture
from phonetics.evaluation.interpretation_reporting import build_interpretation_presentation_data, write_interpretation_presentation_data
from phonetics.visualization import load_protocol,apply_style,export_figure
from phonetics.visualization.interpretation import build_anchor_forest,build_residual_qq
from phonetics.visualization.interpretation_report import write_interpretation_report


class InterpretationFigureTests(unittest.TestCase):
    def setUp(self):
        self.protocol=load_protocol();apply_style(self.protocol)
        self.prepared=build_interpretation_presentation_data(*paired_fixture())

    def tearDown(self):
        plt.close('all')

    def test_forest_positions_sd_reference_and_fullmatch(self):
        fig=build_anchor_forest(self.prepared['q_groups']['forest'],self.protocol)
        self.assertEqual(len(fig.axes),3)
        labels=[ax.get_yticklabels()[0].get_text() for ax in fig.axes]
        self.assertEqual(labels,['000 → 100','000 → 010','000 → 001'])
        for ax in fig.axes:
            self.assertEqual(len(ax.containers),20)
            self.assertEqual(len({float(c.lines[0].get_ydata()[0]) for c in ax.containers}),20)
            self.assertTrue(any(list(line.get_xdata())==[.5,.5] for line in ax.lines))
            self.assertTrue(any(list(line.get_ydata())==[18.4,18.4] for line in ax.lines))
        self.assertFalse(fig._phonetics_resolution['mc_se_in_figure'])

    def test_qq_order_shared_axes_seed_counts_and_reference(self):
        fig=build_residual_qq(self.prepared['residual_qq'],self.protocol)
        self.assertEqual([ax.get_title() for ax in fig.axes],['Syllable','Components','Acoustic','Combined λ1','Combined λ10','DIMSIM'])
        self.assertEqual([len(ax.lines) for ax in fig.axes],[6,6,6,6,6,2])
        for ax in fig.axes:
            self.assertEqual(ax.get_xlim(),fig.axes[0].get_xlim())
            self.assertEqual(ax.get_ylim(),fig.axes[0].get_ylim())
            np.testing.assert_array_equal(ax.lines[-1].get_xdata(),ax.lines[-1].get_ydata())
            self.assertEqual(ax.lines[-1].get_linestyle(),'--')
            self.assertEqual(ax.get_xlabel(),'Normal quantile')
            self.assertEqual(ax.get_ylabel(),'Standardized residual')

    def test_real_exports_and_no_refit_on_render(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);identity=root/'input.json';identity.write_text('{}')
            numerical=write_interpretation_presentation_data(root,'prepared',self.prepared,
                input_roots=[identity],source_hashes={'source':'d'*64})
            with patch('phonetics.evaluation.interpretation.categorical_fit',side_effect=AssertionError('refit')):
                report=write_interpretation_report(root,'report',prepared_output=numerical,
                    source_hashes={'source':'d'*64},source_digest='e'*64,protocol=self.protocol)
            self.assertEqual(json.loads((report/'manifest.json').read_text())['status'],'complete')
            for suffix in ('png','eps','metadata.json'):
                self.assertTrue((report/'figures'/('residual-qq.'+suffix)).stat().st_size>0)
            self.assertEqual(len(list((report/'tables').glob('*.md'))),9)
            with self.assertRaises(FileExistsError):
                write_interpretation_report(root,'report',prepared_output=numerical,
                    source_hashes={'source':'d'*64},source_digest='e'*64,protocol=self.protocol)
            forest=build_anchor_forest(self.prepared['q_groups']['forest'],self.protocol)
            export_figure(forest,root/'forest','forest','anchor_q_transitions',self.protocol,
                          scientific_provenance={'source_digest':'e'*64})
            self.assertTrue((root/'forest/forest.eps').stat().st_size>0)


if __name__=='__main__':unittest.main()
