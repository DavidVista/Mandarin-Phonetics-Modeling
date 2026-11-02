from __future__ import annotations
import sys, unittest
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import numpy as np
ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"src"))
from phonetics.visualization import load_protocol
from phonetics.visualization.evaluation import MODEL_LABELS
from phonetics.evaluation.evaluation_tables import render_evaluation_tables
from phonetics.visualization.evaluation import build_figures
from phonetics.evaluation.anchor_sampling import EDGES
from phonetics.evaluation.contextual_dimsim import BIN_LABELS, MODES
from phonetics.visualization.evaluation import _check_paired_outputs, _guard_not_in_immutable_run, render_evaluation
from unittest.mock import patch

class EvaluationFigureProtocolTests(unittest.TestCase):
    def test_five_approved_names_and_semantics(self):
        p=load_protocol(); names={p.figure(x)["name"] for x in ("anchor_q_transitions","transition_shares","component_shares","dimsim_geometry","dimsim_correlation")}
        self.assertEqual(names,{"Anchor-Q transitions","Transition shares","Component shares","DIMSIM geometry","DIMSIM correlation"})
        self.assertEqual(len(MODEL_LABELS),5)
        for model,label in MODEL_LABELS.items(): self.assertEqual(p.registry["semantics"][model]["label"],label)

    def test_tables_are_plain_markdown_views(self):
        import tempfile
        q={k:[] for k in ("model_seed_edges","model_seed_metrics","global_metric_summary","paired_deltas","transition_shares","component_shares")}
        d={k:[] for k in ("seed_rho","model_rho","paired_rho_deltas","seed_boundary","bins")}
        with tempfile.TemporaryDirectory() as td:
            paths=render_evaluation_tables(q,d,Path(td));self.assertEqual(len(paths),15)
            self.assertIn("Exploratory boundary",(Path(td)/"seed_boundary.md").read_text())

    def test_immutable_run_ancestor_guard_leaves_no_partial_output(self):
        import json, tempfile
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)/"run";root.mkdir();(root/"manifest.json").write_text(json.dumps({"scope":"numerical_outputs","analysis":"contextual-anchor-q","status":"complete"}))
            nested=root/"figures"/"new";nested.parent.mkdir()
            with self.assertRaisesRegex(ValueError,"immutable"): _guard_not_in_immutable_run(nested)
            self.assertFalse(nested.exists())
            q={k:[] for k in ("model_seed_edges","model_seed_metrics","global_metric_summary","paired_deltas","transition_shares","component_shares")}
            d={k:[] for k in ("seed_rho","model_rho","paired_rho_deltas","seed_boundary","bins")}
            with self.assertRaisesRegex(ValueError,"immutable"):
                render_evaluation_tables(q,d,nested)
            self.assertFalse(nested.exists())

    def test_paired_output_identity_is_keyed_and_roles_are_checked(self):
        def pins(role):
            return {"analysis":role,"bundle_manifest_sha256":"a"*64,"model_ids":["m"],"seeds":[0],"model_seed_provenance":[{"model_id":"m","seed":0,"checkpoint_sha256":"b"*64}]}
        q={"manifest":{"analysis":"contextual-anchor-q","scientific_input_pins":pins("contextual-anchor-q")}}
        d={"manifest":{"analysis":"contextual-dimsim","scientific_input_pins":pins("contextual-dimsim")}}
        q["manifest"]["scientific_input_pins"]["model_seed_provenance"].reverse()
        _check_paired_outputs(q,d)
        d["manifest"]["scientific_input_pins"]["bundle_manifest_sha256"]="c"*64
        with self.assertRaisesRegex(ValueError,"bundle_manifest_sha256"): _check_paired_outputs(q,d)

    def test_render_wrapper_uses_real_immutable_loaders(self):
        import tempfile
        sys.path.insert(0,str(ROOT/"tests"))
        import test_evaluation_outputs as fixture_module
        fixture=fixture_module.EvaluationOutputsTests();fixture.setUp()
        try:
            paths=fixture._write();out=fixture.root/"review-figures"
            with patch("phonetics.visualization.evaluation.build_figures",return_value=[]), patch("phonetics.visualization.evaluation.export_figure",return_value=[]):
                self.assertEqual(render_evaluation(paths["contextual-anchor-q"],paths["contextual-dimsim"],out,"d"*64),[])
                self.assertFalse(out.exists())
            guarded=paths["contextual-anchor-q"]/"figures"/"child"
            with self.assertRaisesRegex(ValueError,"immutable"):
                with patch("phonetics.visualization.evaluation.build_figures",return_value=[]):
                    render_evaluation(paths["contextual-anchor-q"],paths["contextual-dimsim"],guarded,"d"*64)
            self.assertFalse(guarded.exists())
            with self.assertRaisesRegex(ValueError,"source_digest"):
                render_evaluation(paths["contextual-anchor-q"],paths["contextual-dimsim"],out,"bad")
        finally: fixture.tearDown()

    def test_nonempty_five_model_table_shapes_counts_and_coverage(self):
        import tempfile
        models=list(MODEL_LABELS);metrics=("initial","final","tone","overall")
        q={k:[] for k in ("model_seed_edges","model_seed_metrics","model_metric_seed_summary","model_edge_seed_summary","global_metric_summary","paired_deltas","transition_shares","component_shares")}
        for mi,model in enumerate(models):
            for seed in range(5):
                for ei,(component,low,high) in enumerate(EDGES):
                    q["model_seed_edges"].append({"model_id":model,"seed":seed,"edge_id":f"{component}:{low}->{high}","q":.5+mi*.01,"mc_se":None if seed==0 else .01,"draw_count":1000,"coverage":{"unique_triplets":17,"unique_anchor_occurrences":9,"unique_anchor_syllables":5,"unique_occurrences":20}})
                for metric in metrics: q["model_seed_metrics"].append({"model_id":model,"seed":seed,"metric":metric,"q":.5+mi*.01,"mc_se":.01})
            for ei,(component,low,high) in enumerate(EDGES): q["model_edge_seed_summary"].append({"model_id":model,"edge_id":f"{component}:{low}->{high}","mean":.5,"sample_sd":.02,"mc_se":.01})
            for metric in metrics:
                q["model_metric_seed_summary"].append({"model_id":model,"metric":metric,"mean":.5,"sample_sd":.02,"mc_se":.01})
                q["global_metric_summary"].append({"model_id":model,"metric":metric,"mean":.5,"sample_sd":.02,"mc_se":.01})
            for edge in EDGES:
                component,low,high=edge;q["transition_shares"].append({"model_id":model,"edge_id":f"{component}:{low}->{high}","share":1/12,"status":"ok"})
            for component in ("initial","final","tone"): q["component_shares"].append({"model_id":model,"component":component,"share":1/3,"status":"ok"})
            if mi:
                for metric in metrics: q["paired_deltas"].append({"model_id":model,"reference_model":models[0],"metric":metric,"mean_delta_q":.01,"sample_sd_seed_delta":.02,"mc_se":.01})
        d={"seed_rho":[],"model_rho":[],"paired_rho_deltas":[],"seed_boundary":[],"model_boundary":[],"bins":[]}
        for mi,model in enumerate(models):
            for mode in MODES:
                for seed in range(5):
                    d["seed_rho"].append({"model_id":model,"seed":seed,"mode":mode,"rho":None if seed==0 else -.2,"valid_draw_count":100,"unique_pair_count":100,"status":"undefined_rho" if seed==0 else "ok"})
                    d["seed_boundary"].append({"model_id":model,"seed":seed,"mode":mode,"threshold":None if seed==0 else 55,"status":"insufficient" if seed==0 else "ok"})
                d["model_rho"].append({"model_id":model,"mode":mode,"mean":-.2,"sample_sd":.02,"status":"ok"})
                d["model_boundary"].append({"model_id":model,"mode":mode,"mean":55,"sample_sd":5,"median":55,"status":"ok"})
                if mi: d["paired_rho_deltas"].append({"model_id":model,"reference_model":models[0],"mode":mode,"mean":.01,"sample_sd":.02,"status":"ok"})
                for label in BIN_LABELS: d["bins"].append({"model_id":model,"mode":mode,"bin":label,"mean_cosine":None,"supported":False,"seed_points":[{"draw_count":100,"unique_pair_count":100}]})
        with tempfile.TemporaryDirectory() as td:
            paths=render_evaluation_tables(q,d,Path(td))
            self.assertEqual(len([p for p in paths if p.name.startswith("anchor_q_edges_") and "_seed_" in p.name]),25)
            edge_text=(Path(td)/f"anchor_q_edges_{models[0]}_seed_0.md").read_text(encoding="utf-8");self.assertIn(chr(0x2014).encode("utf-8"),edge_text.encode("utf-8"));self.assertIn("| 1000 | 17 | 9 | 5 | 20 |",edge_text)
            global_lines=(Path(td)/"anchor_q_global_summary_wide.md").read_text().splitlines();self.assertEqual(sum(line.startswith("| ") for line in global_lines[3:]),5)
            paired_lines=(Path(td)/"anchor_q_paired_deltas_wide.md").read_text().splitlines();self.assertEqual(sum(line.startswith("| ") for line in paired_lines[3:]),4)
            self.assertIn("| initial | final | tone | overall |",(Path(td)/f"anchor_q_metrics_wide_{models[0]}.md").read_text())
            self.assertEqual(sum(line.startswith("| ") for line in (Path(td)/"dimsim_model_rho_wide.md").read_text().splitlines()[3:]),5)

    def test_full_five_model_mock_builds_five_figures_and_geometry_gaps(self):
        import matplotlib.pyplot as plt
        models=list(MODEL_LABELS)
        aq={"edges":[],"transition_shares":[],"component_shares":[]}
        for mi,model in enumerate(models):
            for ei,(component,low,high) in enumerate(EDGES):
                value=.2+.02*mi+.01*ei
                aq["edges"].append({"model_id":model,"edge_id":f"{component}:{low}->{high}","mean":value,
                    "seed_points":[{"seed":s,"q":value+.005*(s-2)} for s in range(5)]})
                aq["transition_shares"].append({"model_id":model,"edge_id":f"{component}:{low}->{high}","share":(ei+1)/78,"status":"ok"})
            for component in ("initial","final","tone"):
                aq["component_shares"].append({"model_id":model,"component":component,"share":1/3,"status":"ok"})
        ds={"curves":[],"boundaries":[],"mode_counts":[],"rho":[]}
        for mode in MODES:
            natural=mode==MODES[0];ds["mode_counts"].append({"mode":mode,"draw_count":100,"unique_pair_count":100 if natural else 90})
            for mi,model in enumerate(models):
                for bi,label in enumerate(BIN_LABELS):
                    unique=99 if bi in (1,5,6) else (100 if natural else 90);draws=unique if natural else 100
                    supported=unique>=100; value=-.35 if mi==4 else .8-.03*bi+.01*mi
                    ds["curves"].append({"model_id":model,"mode":mode,"bin":label,"mean_cosine":value,
                        "supported":supported,"connect_to_next":supported and bi+1 not in (1,5,6),
                        "seed_points":[{"seed":s,"mean_cosine":value+(s-2)*.005,"draw_count":draws,"unique_pair_count":unique} for s in range(5)]})
                ds["boundaries"].append({"model_id":model,"mode":mode,"median":None if mi==0 and natural else 40.,"seed_points":[]})
            ds["rho"].extend({"model_id":model,"mode":mode,"mean":-.2+.05*mi,"seed_values":[{"seed":s,"rho":-.2+.05*mi} for s in range(5)]} for mi,model in enumerate(models))
        figs=build_figures(aq,ds)
        try:
            self.assertEqual(len(figs),5)
            self.assertEqual(len(figs[3].axes),20)
            for ax in figs[3].axes[10:]: self.assertEqual(list(ax.get_yticks()),[0,100])
            host=figs[3].axes[2];segments=[tuple(line.get_xdata()) for line in host.lines if len(line.get_xdata())==2]
            self.assertFalse(any(min(x)<=4 and max(x)>=7 for x in segments))
            self.assertTrue(any(tuple(x)==(3,3.5) for x in segments));self.assertTrue(any(tuple(x)==(3.5,4) for x in segments))
            self.assertEqual(host.get_ylim(),(-1.0,1.0))
            self.assertEqual([figs[3].axes[10].patches[0].get_height(),figs[3].axes[10].patches[13].get_height()],[100,100])
            self.assertEqual([figs[3].axes[11].patches[0].get_height(),figs[3].axes[11].patches[13].get_height()],[100,90])
            self.assertEqual(figs[3].axes[0].get_ylabel(),"Cosine similarity")
            self.assertEqual(figs[3].axes[1].get_ylabel(),"")
            self.assertTrue(any(line.get_marker()=="D" and np.isclose(line.get_xdata()[0],3.5) for line in host.lines))
            self.assertFalse(any(line.get_marker()=="D" for line in figs[3].axes[0].lines))
            partial=[(type(a).__name__,a.get_alpha()) for f in figs for a in f.findobj() if hasattr(a,"get_alpha") and (not hasattr(a,"get_visible") or a.get_visible()) and a.get_alpha() is not None and 0<a.get_alpha()<1]
            self.assertEqual(partial,[])
        finally:
            for fig in figs: plt.close(fig)

if __name__=="__main__": unittest.main()
