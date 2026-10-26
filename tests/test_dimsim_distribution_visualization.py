import importlib.util
import json
from pathlib import Path
import tempfile
import tomllib
import unittest
import numpy as np

PROJECT=Path(__file__).parents[1]
HAS_MATPLOTLIB=importlib.util.find_spec("matplotlib") is not None

class DimsimDistributionVisualizationTests(unittest.TestCase):
    def test_distribution_palette_is_grayscale(self):
        registry=tomllib.loads((PROJECT/"src/phonetics/visualization/registry.toml").read_text(encoding="utf-8"))
        for key in ("000","001","010","011","100","101","110","111","initial","final","tone"):
            color=registry["semantics"][key]["color"].lstrip("#"); rgb=tuple(int(color[i:i+2],16) for i in (0,2,4)); self.assertEqual(rgb,(rgb[0],)*3,key)

    @unittest.skipUnless(HAS_MATPLOTLIB,"matplotlib unavailable")
    def test_wide_layout_aspect_labels_and_all_exports(self):
        import matplotlib.pyplot as plt
        from phonetics.visualization import apply_style,layout,load_protocol
        from phonetics.visualization.distribution import render_distribution
        protocol=load_protocol(); apply_style(protocol)
        fig,axes=layout(1,protocol,legend_rows=1,width_mm=protocol.style["layout"]["multi_width_mm"])
        ax=axes[0]; labels=[f"{10*i}\u2013{10*(i+1)}" for i in range(12)]+["\u2265120"]; ax.set_xticks(np.arange(13),labels); fig.canvas.draw()
        width,height=ax.get_window_extent().width,ax.get_window_extent().height; self.assertAlmostEqual(width/height,protocol.style["layout"]["panel_aspect"],places=10)
        boxes=[label.get_window_extent(fig.canvas.get_renderer()) for label in ax.get_xticklabels()]
        self.assertTrue(all(a.x1<=b.x0 for a,b in zip(boxes,boxes[1:]))); plt.close(fig)
        temp_parent=Path(__import__("os").environ.get("TEMP",PROJECT));
        with tempfile.TemporaryDirectory(dir=temp_parent) as tmp:
            root=Path(tmp); (root/"data").mkdir(); (root/"figures").mkdir(); d=np.array([.1,2,8,12,25,45,65,85,105,115,125.]); masks=np.array(["000","001","010","011","100","101","110","111","000","001","010"])
            np.savez_compressed(root/"data/distribution_plot_data.npz",distance=d,overlap_mask=masks,detail_values=d[d<10],full_histogram_edges=np.r_[np.arange(0.,121.,10.),125.000001],initial_contribution=d*.3,final_contribution=d*.6,tone_contribution=d*.1)
            rows=[{"bin":f"[{i},{i+10})" if i<120 else "[120,+inf)","count":1,"mean_distance":1.,"mean_initial":.3,"mean_final":.6,"mean_tone":.1,"initial_share":.3,"final_share":.6,"tone_share":.1} for i in range(0,121,10)]; (root/"data/component_shares.json").write_text(json.dumps(rows),encoding="utf-8")
            exported=render_distribution(root/"data",root/"figures","a"*64); self.assertEqual(len(exported),6); self.assertEqual(len(list((root/"figures").iterdir())),9); self.assertEqual(plt.get_fignums(),[])

if __name__=="__main__": unittest.main()
