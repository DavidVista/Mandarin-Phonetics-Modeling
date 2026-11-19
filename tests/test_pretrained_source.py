"""Pretrained exports are a single pinned source, not fabricated training runs."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from phonetics.artifacts import write_json
from phonetics.bundles import seal_bundle
from phonetics.encoder_jobs import locate_exports, resolve_encoder_config, validate_run_inventory
from phonetics.evaluation.encoder_extraction import load_export
from phonetics.evaluation.pretrained_source import TYPE
from phonetics.research.encoder_geometry import _summarize_seed_tables


class PretrainedSourceTests(unittest.TestCase):
    def test_single_array_reused_with_checked_snapshot_and_unknown_positions(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/"pretrained-bert"; root.mkdir()
            write_json(root/"occurrence_ids.json",["sample:0","sample:1"])
            np.save(root/"hidden.npy",np.ones((2,768),dtype=np.float32))
            np.save(root/"unknown.npy",np.array([False,True]))
            identity=dict(model="pretrained-bert",evaluation_manifest_sha256="a"*64,
                experiment_manifest_sha256="b"*64,source_manifest_sha256="b"*64,
                extraction_count=1,representation="last layer pre-head",snapshot={"revision":"pinned"})
            seal_bundle(root,TYPE,identity)
            found=locate_exports({"representations_root":folder},
                {"evaluation":"a"*64,"experiment":"b"*64,"pretrained-bert":"b"*64},["pretrained-bert"])
            self.assertEqual(found,{"pretrained-bert":root.resolve()})
            args=dict(model="pretrained-bert",evaluation_pin="a"*64,experiment_pin="b"*64,
                training_pin="b"*64,occurrence_ids=["sample:0","sample:1"],
                expected_seeds={s:{"snapshot":{"revision":"pinned"},"seed_input_hashes":{}} for s in (0,1,2)},
                expected_unknown={s:np.array([False,True]) for s in (0,1,2)},
                expected_input_hashes={s:{} for s in (0,1,2)})
            rows=list(load_export(root,**args))
            self.assertEqual([r[0] for r in rows],[0,1,2])
            self.assertIs(rows[0][1],rows[2][1])
            self.assertFalse(rows[1][3]["independent_training_seed"])
            args["expected_seeds"][1]["snapshot"]={"revision":"wrong"}
            with self.assertRaisesRegex(ValueError,"source or inputs"):
                list(load_export(root,**args))

    def test_source_config_and_explicit_subset(self):
        spec=dict(id="report",training_runs={},representations_root="hidden",device="cuda",batch_size=8,
            tolerance=1e-12,representation_sources={"pretrained-bert":{"type":"pretrained_bert"}})
        self.assertEqual(resolve_encoder_config(spec,str),spec)
        experiment={"identity":{"config":{"model_ids":["components","acoustic"],
                                            "architecture":{"encoder":{"type":"bert"}}}}}
        self.assertEqual(validate_run_inventory({"experiment":"x","evaluation":"y","components":"z"},experiment),["components"])
        self.assertEqual(validate_run_inventory({"experiment":"x","evaluation":"y","pretrained-bert":"x"},experiment),["pretrained-bert"])

    def test_deterministic_source_has_no_independent_seed_sd(self):
        from collections import defaultdict
        tables=defaultdict(list)
        tables["mask_profiles"]=[dict(seed=s,system="pretrained-bert",mask="000",mean_similarity=.4,pairs=5) for s in (0,1,2)]
        row=_summarize_seed_tables(tables,[0,1,2])["mask_profiles"][0]
        self.assertIsNone(row["mean_similarity_sd"])
        self.assertEqual(row["mean_similarity_seeds"],1)


if __name__=="__main__":
    unittest.main()
