"""Thin orchestration for the retained DIMSIM distribution analysis."""
from __future__ import annotations
import argparse, csv, json
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
from .artifacts import file_sha256, research_run
from ..evaluation.distribution import compute_retained_population, retained_distribution_data, summarize_population
from ..visualization.distribution import render_distribution

@dataclass(frozen=True)
class DimsimDistributionConfig:
    project_root: Path; run_id: str; evaluation_root: Path; evaluation_manifest_sha256: str
    def resolved(self): return {k: str(v.resolve()) if isinstance(v,Path) else v for k,v in asdict(self).items()}

def run_dimsim_distribution(config: DimsimDistributionConfig):
    package=Path(__file__).parents[1]; viz=package/"visualization"
    sources=[Path(__file__),Path(__file__).with_name("artifacts.py"),package/"evaluation"/"distribution.py",package/"evaluation"/"dimsim.py",package/"evaluation"/"geometry_bundle.py",package/"bundles.py",viz/"core.py",viz/"distribution.py",viz/"style.toml",viz/"registry.toml"]
    expected={"config.json","data/distribution_plot_data.npz","data/component_shares.json","tables/distribution_bins.csv","tables/overlap_masks.csv","tables/component_shares.csv","diagnostics/validation.json","figures/dimsim-distributions.png","figures/dimsim-distributions.eps","figures/dimsim-distributions.metadata.json","figures/component-overlap-distributions.png","figures/component-overlap-distributions.eps","figures/component-overlap-distributions.metadata.json","figures/dimsim-component-contributions.png","figures/dimsim-component-contributions.eps","figures/dimsim-component-contributions.metadata.json"}
    expected.add("tables/descriptive_summary.json")
    with research_run(config.project_root,"dimsim-distribution",config.run_id,[config.evaluation_root],config.resolved(),sources,expected) as (root,manifest):
        columns,validation=compute_retained_population(config.evaluation_root,expected_manifest_sha256=config.evaluation_manifest_sha256)
        plot,rows=retained_distribution_data(columns); np.savez_compressed(root/"data/distribution_plot_data.npz",**plot)
        (root/"data/component_shares.json").write_text(json.dumps(rows,indent=2,allow_nan=False),encoding="utf-8")
        def write_csv(name,fieldnames,records):
            with (root/"tables"/name).open("x",encoding="utf-8",newline="") as stream:
                writer=csv.DictWriter(stream,fieldnames=fieldnames); writer.writeheader(); writer.writerows(records)
        distribution_rows=[]
        for row in rows:
            low=120 if row["bin"]=="[120,+inf)" else int(row["bin"].split(",")[0][1:]); selected=plot["distance"]>=low if low==120 else ((plot["distance"]>=low)&(plot["distance"]<low+10)); count=int(selected.sum())
            distribution_rows.append({"bin":row["bin"],"count":count,"share":count/len(plot["distance"])})
        mask_rows=[{"mask":m,"count":int((plot["overlap_mask"]==m).sum()),"share":float((plot["overlap_mask"]==m).mean())} for m in (f"{i:03b}" for i in range(8))]
        write_csv("distribution_bins.csv",["bin","count","share"],distribution_rows); write_csv("overlap_masks.csv",["mask","count","share"],mask_rows); write_csv("component_shares.csv",list(rows[0]),rows)
        validation.update({"overlap_mask_counts":{m:int((plot["overlap_mask"]==m).sum()) for m in (f"{i:03b}" for i in range(8))},"formula_reconstruction_max_absolute_error":float(np.max(np.abs(plot["initial_contribution"]+plot["final_contribution"]+plot["tone_contribution"]-plot["distance"])))})
        (root/"diagnostics/validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
        summary = summarize_population(columns, validation)
        (root/"tables/descriptive_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        data_hash=file_sha256(root/"data/distribution_plot_data.npz"); render_hash=file_sha256(viz/"distribution.py"); render_distribution(root/"data",root/"figures",render_hash)
        manifest["scientific_provenance"]={"population":"complete unordered pairs on the selected evaluation inventory","pair_count":len(plot["distance"]),"evaluation_manifest_sha256":config.evaluation_manifest_sha256,"metric_provenance":validation["metric_provenance"],"plot_data_sha256":data_hash,"figure_source_sha256":render_hash,"intervals":"left-closed 10-unit bins with >=120 overflow","overlap_masks":"exact initial/final/tone equality bits"}; return root

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--project-root",type=Path,required=True); p.add_argument("--run-id",required=True); p.add_argument("--evaluation-root",type=Path,required=True); p.add_argument("--evaluation-manifest-sha256",required=True); a=p.parse_args(argv)
    run_dimsim_distribution(DimsimDistributionConfig(a.project_root,a.run_id,a.evaluation_root,a.evaluation_manifest_sha256))
if __name__=="__main__": main()
