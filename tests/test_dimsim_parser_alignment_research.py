import csv,gzip,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import test_dimsim_alignment as legacy
from phonetics.evaluation import dimsim_alignment as evaluation
from phonetics.research.dimsim_parser_alignment import DimsimParserAlignmentConfig,run_dimsim_parser_alignment

def fixture_results():
    rows=evaluation.build_decompositions(legacy.bundle(),author_parser=legacy.parser(legacy.A),strict_parser=legacy.parser(legacy.B)); order={label:i for i,label in enumerate(sorted(legacy.A))}; score=lambda left,right:float(order[left]+order[right])
    reproduce=lambda left,right:score(next(label for label,obj in legacy.A.items() if obj is left),next(label for label,obj in legacy.A.items() if obj is right))
    alignment=evaluation.evaluate_alignment(rows,scorer_a=score,scorer_b=lambda a,b:score(a,b)+2,reproduction_scorer=reproduce,author_parser=legacy.parser(legacy.A)); statistics=evaluation.summarize_statistics(rows,alignment,seed=42,bootstrap_replicates=5,confidence=.95,atol=1e-12,rtol=1e-10); return rows,alignment,statistics

class DimsimParserAlignmentResearchTests(unittest.TestCase):
    def test_structured_writer_is_numerically_identical_to_legacy_export(self):
        rows,alignment,statistics=fixture_results()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); shared=root/"shared"; legacy_root=root/"legacy"; (shared/"tables").mkdir(parents=True); (shared/"diagnostics").mkdir()
            evaluation.write_alignment_research_artifacts(shared,decompositions=rows,alignment=alignment,statistics=statistics,validation={"ok":True})
            evaluation.export_alignment(legacy_root,config={},decompositions=rows,alignment=alignment,statistics=statistics,provenance={})
            mapping={"decompositions.csv":"decompositions.csv","coverage.csv":"coverage.csv","pairs.csv.gz":"pairs.csv.gz","symmetry-summary.csv":"symmetry-summary.csv","symmetry-exceptions.csv.gz":"symmetry-exceptions.csv.gz","statistics-by-syllable.csv":"statistics-by-syllable.csv","statistics-by-category.csv":"statistics-by-category.csv"}
            for new,old in mapping.items():
                opener=gzip.open if new.endswith(".gz") else open
                with opener(shared/"tables"/new,"rt",encoding="utf-8") as a,opener(legacy_root/old,"rt",encoding="utf-8") as b:self.assertEqual(a.read(),b.read(),new)
            self.assertEqual(json.loads((shared/"diagnostics/bootstrap-diagnostics.json").read_text()),json.loads((legacy_root/"bootstrap-diagnostics.json").read_text()))

    def test_config_requires_explicit_hash_and_preserves_protocol_defaults(self):
        config=DimsimParserAlignmentConfig(Path("."),"run",Path("input"),"a"*64)
        self.assertEqual((config.seed,config.bootstrap_replicates,config.confidence,config.atol,config.rtol),(42,1000,.95,1e-12,1e-10))
        with self.assertRaisesRegex(ValueError,"required"):run_dimsim_parser_alignment(DimsimParserAlignmentConfig(Path("."),"run",Path("input"),""))


if __name__=="__main__":unittest.main()
