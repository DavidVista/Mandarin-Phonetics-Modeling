import csv,gzip,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import test_dimsim_alignment as legacy
from phonetics.evaluation import dimsim_alignment as evaluation
from phonetics.research.dimsim_parser_alignment import DimsimParserAlignmentConfig,run_dimsim_parser_alignment
from phonetics.research.artifacts import file_sha256
import test_character_experiment as character_experiment

def fixture_results():
    rows=evaluation.build_decompositions(legacy.bundle(),author_parser=legacy.parser(legacy.A),strict_parser=legacy.parser(legacy.B)); order={label:i for i,label in enumerate(sorted(legacy.A))}; score=lambda left,right:float(order[left]+order[right])
    reproduce=lambda left,right:score(next(label for label,obj in legacy.A.items() if obj is left),next(label for label,obj in legacy.A.items() if obj is right))
    alignment=evaluation.evaluate_alignment(rows,scorer_a=score,scorer_b=lambda a,b:score(a,b)+2,reproduction_scorer=reproduce,author_parser=legacy.parser(legacy.A)); statistics=evaluation.summarize_statistics(rows,alignment,seed=42,bootstrap_replicates=5,confidence=.95,atol=1e-12,rtol=1e-10); return rows,alignment,statistics

class DimsimParserAlignmentResearchTests(unittest.TestCase):
    def test_exclusion_diagnostics_include_exact_labels_and_require_complete_support(self):
        rows, alignment, _ = fixture_results()
        by_label = {row["syllable"]: row for row in alignment["exclusion_diagnostics"]}
        self.assertEqual(by_label["ma1"]["category"], "exact")
        self.assertEqual(by_label["ma1"]["mean_absolute_delta"], 2)
        self.assertEqual(by_label["ma1"]["valid_partners"], 3)
        self.assertFalse(by_label["ma1"]["above_historical_threshold"])
        incomplete = evaluation.evaluate_alignment(
            rows, scorer_a=lambda *_: 0, scorer_b=lambda *_: -1,
            reproduction_scorer=lambda *_: 0, author_parser=legacy.parser(legacy.A),
        )
        self.assertTrue(all(row["above_historical_threshold"] is None for row in incomplete["exclusion_diagnostics"]))

    def test_character_experiment_uses_only_common_test_inventory(self):
        fixture = character_experiment.CharacterExperimentTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        experiment = fixture.build()
        config = DimsimParserAlignmentConfig(
            project_root=fixture.root, run_id="test-inventory", experiment_root=experiment,
            experiment_manifest_sha256=file_sha256(experiment / "manifest.json"), bootstrap_replicates=2,
        )
        with patch("phonetics.research.dimsim_parser_alignment.load_evaluation_bundle",
                   side_effect=AssertionError("experiment input must not load an evaluation bundle")):
            output = run_dimsim_parser_alignment(config)
        split_ids = json.loads((experiment / "splits.json").read_text())["test"]
        corpus = [json.loads(line) for line in (experiment / "corpus.jsonl").read_text(encoding="utf-8").splitlines()]
        expected = {label for row in corpus if row["sample_id"] in split_ids for label in row["labels"]}
        with (output / "tables/decompositions.csv").open(encoding="utf-8", newline="") as stream:
            self.assertEqual({row["label"] for row in csv.DictReader(stream)}, expected)
        validation = json.loads((output / "diagnostics/validation.json").read_text())
        self.assertEqual((validation["split"], validation["sample_count"]), ("test", len(split_ids)))
        provenance = json.loads((output / "manifest.json").read_text())["scientific_provenance"]
        self.assertIn("affected_union_fraction", provenance)
        self.assertIs(provenance["use_both_directions"], False)
        self.assertTrue((output / "tables/exclusion-diagnostics.csv").is_file())

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
