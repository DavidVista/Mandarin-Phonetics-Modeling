from __future__ import annotations
import importlib.util, json, sys, tempfile, unittest
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PKG=Path(__file__).parents[1]/"src"/"phonetics"/"visualization"
spec=importlib.util.spec_from_file_location("viz_under_test",PKG/"__init__.py",submodule_search_locations=[str(PKG)])
viz=importlib.util.module_from_spec(spec); sys.modules[spec.name]=viz; spec.loader.exec_module(viz)

def protocol(): return viz.load_protocol(PKG/"style.toml",PKG/"registry.toml")

def approved(td):
    s=td/"style.toml";r=td/"registry.toml";s.write_bytes((PKG/"style.toml").read_bytes());r.write_text((PKG/"registry.toml").read_text().replace('status = "draft"','status = "approved"',1));return viz.load_protocol(s,r)

class Tests(unittest.TestCase):
    def setUp(self):
        existing = set(plt.get_fignums())
        self.addCleanup(self.close_new_figures, existing)

    def close_new_figures(self, existing):
        for number in set(plt.get_fignums()) - existing:
            plt.close(number)

    def test_schema_layout_and_math(self):
        with tempfile.TemporaryDirectory() as x:
            d=Path(x);r=d/"r";r.write_bytes((PKG/"registry.toml").read_bytes())
            for old,new in [("tick_pt = 8.0","tick_pt = -1.0"),("dpi = 300","dpi = 3"),("minor_ticks = false",'minor_ticks = "no"')]:
                s=d/"s";s.write_text((PKG/"style.toml").read_text().replace(old,new));self.assertRaises(viz.VisualizationError,viz.load_protocol,s,r)
        p=protocol();fig,axes=viz.layout(3,p);self.assertEqual(len(fig.axes),3);self.assertAlmostEqual(axes[0].get_position().width,axes[2].get_position().width)
        x,y=viz.ecdf([2,1,2]);np.testing.assert_array_equal(x,[1,2,2]);np.testing.assert_allclose(y,[1/3,2/3,1])
        fig,axes=viz.layout(1,p);self.assertRaises(viz.VisualizationError,viz.plot_histogram,axes[0],{"base":[0],"zh_cn":[1]},p.figure("pilot_histogram"),p,edges=[0,1,np.inf])
    def test_kind_schema_counterexamples(self):
        with tempfile.TemporaryDirectory() as x:
            d=Path(x);s=d/"s";s.write_bytes((PKG/"style.toml").read_bytes());base=(PKG/"registry.toml").read_text()
            for changed in [base.replace("ylim = [0.0, 1.0]\n","",1),base.replace("density = true\n","",1),base.replace("xlim = [-1.0, 1.0]","xlim = [1.0, -1.0]",1),base.replace("edges_count = 61","edges_count = true")]:
                r=d/"r";r.write_text(changed);self.assertRaises(viz.VisualizationError,viz.load_protocol,s,r)
    def test_snapshot_external_and_tamper(self):
        with tempfile.TemporaryDirectory() as x:
            d=Path(x);p=approved(d);snap=viz.create_snapshot(d/"snap",p);self.assertIn('status = "approved"',(snap/"registry.toml").read_text());viz.verify_snapshot(snap);(snap/"style.toml").write_text((snap/"style.toml").read_text()+"\n");self.assertRaises(viz.VisualizationError,viz.verify_snapshot,snap)
    def test_export_success_overwrite_alpha(self):
        with tempfile.TemporaryDirectory() as x:
            d=Path(x);p=approved(d);viz.apply_style(p);matplotlib.rcParams["savefig.bbox"]="tight";fig,axes=viz.layout(1,p,legend_rows=1);viz.plot_ecdf(axes[0],{"base":[1,2,3],"zh_cn":[2,3,4]},p.figure("pilot_ecdf"),p);viz.shared_legend(fig,axes,p);out=d/"out";paths=viz.export_figure(fig,out,"x","pilot_ecdf",p,scientific_provenance={"source_digest":"0"*64});self.assertEqual([z.suffix for z in paths],[".png",".eps"]);self.assertTrue(all(z.stat().st_size for z in paths));
            with Image.open(paths[0]) as image: actual=image.size
            expected=tuple(v*300 for v in fig.get_size_inches());self.assertTrue(all(abs(a-e)<=1 for a,e in zip(actual,expected)));self.assertEqual(json.loads((out/"x.metadata.json").read_text())["resolved_data"]["counts"]["base"],3)
            (out/"y.metadata.json").write_text("KEEP");self.assertRaises(FileExistsError,viz.export_figure,fig,out,"y","pilot_ecdf",p,scientific_provenance={"source_digest":"0"*64});self.assertEqual((out/"y.metadata.json").read_text(),"KEEP")
            f,a=viz.layout(1,p);a[0].plot([0,1],[0,1],alpha=.3);self.assertRaises(viz.VisualizationError,viz.export_figure,f,d/"alpha","z","pilot_ecdf",p,scientific_provenance={"source_digest":"0"*64});self.assertFalse((d/"alpha"/"z.png").exists())
            f,a=viz.layout(1,p);a[0].plot([0,1],[0,1],color=(0,0,0,.3));self.assertRaises(viz.VisualizationError,viz.export_figure,f,d/"rgba","z","pilot_ecdf",p,scientific_provenance={"source_digest":"0"*64});self.assertFalse((d/"rgba"/"z.png").exists())
            f,a=viz.layout(1,p);a[0].scatter([0],[0],facecolors=[(0,0,0,.3)]);self.assertRaises(viz.VisualizationError,viz.export_figure,f,d/"scatter","z","pilot_ecdf",p,scientific_provenance={"source_digest":"0"*64})
    def test_visible_tick_clipping(self):
        with tempfile.TemporaryDirectory() as x:
            d=Path(x);p=approved(d);viz.apply_style(p);f,a=viz.layout(1,p);a[0].set_xlim(0,1);a[0].set_xticks([0]);a[0].set_xticklabels(["VISIBLE TICK LABEL THAT IS FAR TOO LONG TO FIT IN THE CANVAS"]);self.assertRaises(viz.VisualizationError,viz.export_figure,f,d/"clip","z","pilot_ecdf",p,scientific_provenance={"source_digest":"0"*64})
    def test_draft_and_mutation(self):
        with tempfile.TemporaryDirectory() as x:
            d=Path(x);style=d/"style.toml";registry=d/"registry.toml"
            style.write_bytes((PKG/"style.toml").read_bytes())
            registry.write_text((PKG/"registry.toml").read_text(encoding="utf-8").replace('status = "approved"','status = "draft"'),encoding="utf-8")
            p=viz.load_protocol(style,registry);self.assertEqual(p.figure("pilot_ecdf")["status"],"draft")
            f,_=viz.layout(1,p)
            with self.assertRaisesRegex(viz.VisualizationError,"draft"):
                viz.export_figure(f,d/"out","draft","pilot_ecdf",p,scientific_provenance={"source_digest":"0"*64})
            self.assertFalse((d/"out").exists())
            p.style["font"]["tick_pt"]=99
            with self.assertRaisesRegex(viz.VisualizationError,"mutated"):
                viz.create_snapshot(d/"snap",p)

if __name__=="__main__":unittest.main()
