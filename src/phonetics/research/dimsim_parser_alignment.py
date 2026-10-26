"""Shared-output orchestration for the DIMSIM parser alignment study."""
from __future__ import annotations
import argparse,hashlib,json,math,re
from dataclasses import asdict,dataclass
from pathlib import Path
from .artifacts import research_run
from ..evaluation.bundle import load_evaluation_bundle
from ..evaluation.dimsim import DimsimDistance
from ..evaluation.dimsim_pypinyin import PypinyinDimsimDistance
from ..evaluation.dimsim_alignment import build_decompositions,eligible_labels,evaluate_alignment,summarize_statistics,write_alignment_research_artifacts

@dataclass(frozen=True)
class DimsimParserAlignmentConfig:
    project_root:Path; run_id:str; evaluation_root:Path; evaluation_manifest_sha256:str
    seed:int=42; bootstrap_replicates:int=1000; confidence:float=.95; atol:float=1e-12; rtol:float=1e-10
    def resolved(self):return {k:str(v.resolve()) if isinstance(v,Path) else v for k,v in asdict(self).items()}

def _validate(config):
    if not isinstance(config.evaluation_manifest_sha256,str) or re.fullmatch(r"[0-9a-fA-F]{64}",config.evaluation_manifest_sha256) is None:raise ValueError("evaluation_manifest_sha256 is required")
    if isinstance(config.seed,bool) or not isinstance(config.seed,int) or config.seed<0:raise ValueError("seed must be a nonnegative integer")
    if isinstance(config.bootstrap_replicates,bool) or not isinstance(config.bootstrap_replicates,int) or config.bootstrap_replicates<1:raise ValueError("bootstrap_replicates must be positive")
    if isinstance(config.confidence,bool) or not isinstance(config.confidence,(int,float)) or not 0<config.confidence<1:raise ValueError("confidence must be between zero and one")
    if any(isinstance(getattr(config,k),bool) or not isinstance(getattr(config,k),(int,float)) or not math.isfinite(getattr(config,k)) or getattr(config,k)<0 for k in ("atol","rtol")):raise ValueError("tolerances must be finite and nonnegative")

def run_dimsim_parser_alignment(config:DimsimParserAlignmentConfig):
    _validate(config); package=Path(__file__).parents[1]; evaluation=package/"evaluation"
    sources=[Path(__file__),Path(__file__).with_name("artifacts.py"),evaluation/"dimsim_alignment.py",evaluation/"dimsim.py",evaluation/"dimsim_pypinyin.py",evaluation/"bundle.py",package/"pinyin.py"]
    table_names=("decompositions.csv","coverage.csv","pairs.csv.gz","symmetry-summary.csv","symmetry-exceptions.csv.gz","statistics-by-syllable.csv","statistics-by-category.csv")
    expected={"config.json",*(f"tables/{name}" for name in table_names),"diagnostics/bootstrap-diagnostics.json","diagnostics/validation.json"}
    with research_run(config.project_root,"dimsim-parser-alignment",config.run_id,[config.evaluation_root],config.resolved(),sources,expected) as (root,manifest):
        manifest_path=Path(config.evaluation_root)/"manifest.json"; actual=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if actual!=config.evaluation_manifest_sha256:raise ValueError("evaluation manifest SHA256 mismatch")
        bundle=load_evaluation_bundle(config.evaluation_root); decompositions=build_decompositions(bundle)
        alignment=evaluate_alignment(decompositions,atol=config.atol,rtol=config.rtol)
        statistics=summarize_statistics(decompositions,alignment,seed=config.seed,bootstrap_replicates=config.bootstrap_replicates,confidence=config.confidence,atol=config.atol,rtol=config.rtol)
        validation={"evaluation_manifest_sha256":actual,"inventory_count":len(decompositions),"eligible_b_nonzero_count":len(eligible_labels(decompositions)),"affected_pair_count":len(alignment["affected_pairs"]),"symmetry_summary":alignment["symmetry_summary"],"figures":[],"plot_data":[]}
        write_alignment_research_artifacts(root,decompositions=decompositions,alignment=alignment,statistics=statistics,validation=validation)
        manifest["scientific_provenance"]={"evaluation_manifest_sha256":actual,"seed":config.seed,"bootstrap_replicates":config.bootstrap_replicates,"confidence":config.confidence,"atol":config.atol,"rtol":config.rtol,"scorer_a":DimsimDistance().provenance,"scorer_b":PypinyinDimsimDistance().provenance,"figures":[],"plot_data":[],"affected_union_fraction":statistics["affected_union_fraction"],"use_both_directions":statistics["use_both_directions"]};return root

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--project-root",type=Path,required=True);p.add_argument("--run-id",required=True);p.add_argument("--evaluation-root",type=Path,required=True);p.add_argument("--evaluation-manifest-sha256",required=True);a=p.parse_args(argv);run_dimsim_parser_alignment(DimsimParserAlignmentConfig(a.project_root,a.run_id,a.evaluation_root,a.evaluation_manifest_sha256))
if __name__=="__main__":main()
