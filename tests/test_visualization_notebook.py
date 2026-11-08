"""Exercise the artifact-only evaluation workflow directly through the SDK."""
import hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'tests'),str(PROJECT/'src')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from phonetics.visualization import load_protocol
from phonetics.visualization.evaluation_workflow import run_complete_evaluation_report
from phonetics.visualization.evaluation_report import write_evaluation_report

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def render_report(ns):
    roots=ns['SAVED']
    ns['REPORT_PATH']=run_complete_evaluation_report(ns['OUTPUT_PROJECT_ROOT'],ns['PRESENTATION_ID'],
        anchor_q_output=roots['numerical']['contextual-anchor-q'],dimsim_output=roots['numerical']['contextual-dimsim'],
        random_control_output=roots['random_control'],bundle_path=ns['BUNDLE_PATH'],
        bundle_manifest_sha256=ns['BUNDLE_MANIFEST_SHA256'],derived_run_ids=ns['DERIVED_RUN_IDS'],
        derived_mode=ns['DERIVED_MODE'],source_hashes=ns['SOURCE_HASHES'],source_digest=ns['SOURCE_DIGEST'],protocol=ns['PROTOCOL'])

class VisualizationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.ns={'run_complete_evaluation_report':run_complete_evaluation_report,'write_evaluation_report':write_evaluation_report,
                 'plt':plt,'PROTOCOL':load_protocol(),'SOURCE_HASHES':{'test':'a'*64},'SOURCE_DIGEST':'b'*64}

    def test_workflow_rejects_invalid_mode_and_existing_target_before_analysis(self):
        with tempfile.TemporaryDirectory(dir=PROJECT) as directory:
            root=Path(directory)
            inputs=[]
            for name in ('q','dimsim','random','bundle'):
                path=root/name;path.mkdir();inputs.append(path)
            config=dict(project_root=root,presentation_id='report',anchor_q_output=inputs[0],
                dimsim_output=inputs[1],random_control_output=inputs[2],bundle_path=inputs[3],
                bundle_manifest_sha256='a'*64,derived_run_ids={'interpretation':'analysis','presentation_data':'data'},
                derived_mode='invalid',source_hashes={'source':'b'*64},source_digest='c'*64)
            with self.assertRaisesRegex(ValueError,'derived_mode'):
                self.ns['run_complete_evaluation_report'](**config)
            config['derived_mode']='prepare'
            config['derived_run_ids']={}
            with self.assertRaisesRegex(ValueError,'explicit derived run IDs'):
                self.ns['run_complete_evaluation_report'](**config)
            config['derived_run_ids']={'interpretation':'analysis','presentation_data':'data'}
            existing=root/'outputs/contextual-evaluation-report/report';existing.mkdir(parents=True)
            (existing/'keep').write_text('unchanged')
            with self.assertRaises(FileExistsError):
                self.ns['run_complete_evaluation_report'](**config)
            self.assertEqual((existing/'keep').read_text(),'unchanged')
            self.assertFalse((root/'outputs/contextual-interpretation').exists())


    def test_actual_report_from_saved_outputs_without_vector_loading(self):
        import test_evaluation_data_persistence as fixtures
        suite = fixtures.DataPersistenceTests
        suite.setUpClass()
        try:
            case = suite("test_data_stage_needs_no_plotting_settings_or_imports")
            case.setUp()
            saved = case.persist()
            inputs = [*saved["numerical"].values(), saved["random_control"]]
            before = {path: sha(path) for root in inputs for path in root.rglob("*") if path.is_file()}
            ns = self.ns
            ns["SAVED"] = saved
            ns["EVALUATION_OUTPUT_ROOT"] = case.context["project_root"] / "outputs"
            ns["EVALUATION_RUN_IDS"] = None
            ns["OUTPUT_PROJECT_ROOT"] = suite.fixture.root / "visualization-output"
            ns["PRESENTATION_ID"] = "actual-export"
            ns["BUNDLE_PATH"] = case.context["bundle_path"]
            ns["BUNDLE_MANIFEST_SHA256"] = case.context["config"]["bundle_manifest_sha256"]
            ns["DERIVED_MODE"] = "prepare"
            ns["DERIVED_RUN_IDS"] = {"interpretation":"new-analysis","presentation_data":"new-data"}
            with patch("numpy.load", side_effect=AssertionError("Plotting must not load vector matrices")):
                render_report(ns)
            report = ns["REPORT_PATH"]
            for suffix in ("png", "eps", "metadata.json"):
                files = list((report / "figures").glob(f"*.{suffix}"))
                self.assertEqual(len(files), 6)
                self.assertTrue(all(path.stat().st_size > 0 for path in files))
            manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["figure_exports"], "complete")
            for row in manifest["artifacts"]:
                self.assertEqual(sha(report / row["path"]), row["sha256"])
            self.assertEqual(before, {path: sha(path) for path in before})
            self.assertEqual(ns["plt"].get_fignums(), [])
            self.assertTrue((report/'tables/coefficients.md').exists())
            self.assertTrue((report/'data/interpretation_summary.json').exists())
            with self.assertRaises(FileExistsError):
                render_report(ns)
            derived_root = ns["OUTPUT_PROJECT_ROOT"] / 'outputs'
            numerical_roots = [derived_root/'contextual-interpretation/new-analysis', derived_root/'contextual-interpretation-data/new-data']
            hashes = {path:sha(path) for root in numerical_roots for path in root.rglob('*') if path.is_file()}
            ns["DERIVED_MODE"] = 'reuse'
            ns["PRESENTATION_ID"] = 'rerender'
            def forbid_fit(*args, **kwargs):
                raise AssertionError('Reuse must not fit or load raw Q scores')
            with patch('phonetics.visualization.evaluation_workflow.load_saved_q_batch', side_effect=forbid_fit):
                render_report(ns)
            self.assertEqual(hashes,{path:sha(path) for path in hashes})
            self.assertEqual(len(list((ns['REPORT_PATH']/'figures').glob('*.png'))),6)
            altered = suite.fixture.root / "changed/contextual-anchor-q" / saved["numerical"]["contextual-anchor-q"].name
            import shutil
            shutil.copytree(saved["numerical"]["contextual-anchor-q"], altered)
            with (altered / "tables/tables.json").open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(ValueError, "hash or size mismatch"):
                ns["write_evaluation_report"](
                    ns["OUTPUT_PROJECT_ROOT"], "bad-input", anchor_q_output=altered,
                    dimsim_output=saved["numerical"]["contextual-dimsim"],
                    random_control_output=saved["random_control"], source_hashes=ns["SOURCE_HASHES"],
                    source_digest=ns["SOURCE_DIGEST"], protocol=ns["PROTOCOL"],
                )
        finally:
            suite.tearDownClass()
