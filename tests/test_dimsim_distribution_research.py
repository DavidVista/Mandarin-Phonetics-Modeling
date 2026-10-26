import json
import hashlib
from pathlib import Path
import tempfile
import numpy as np
import unittest
from phonetics.evaluation.dimsim import DimsimDistance
from phonetics.evaluation.distribution import compute_retained_population, retained_distribution_data

def fixture_columns():
    return {"distance":np.array([0.,5.,15.,125.]),"overlap_mask":np.array(["111","001","010","000"]),"consonant_distance":np.array([0.,2.,5.,50.]),"vowel_distance":np.array([0.,2.,9.,70.]),"tone_distance":np.array([0.,1.,1.,5.]),"multiplier":np.ones(4),"hardcode_applied":np.zeros(4,dtype=bool)}

def write_bundle(root,*,omit_last=False,bad_mask=False):
    groups=[{"group_id":"a","group_kind":"syllable","pinyin":"ba1","initial":"b","final":"a","tone":"1"},{"group_id":"b","group_kind":"syllable","pinyin":"da2","initial":"d","final":"a","tone":"2"},{"group_id":"c","group_kind":"syllable","pinyin":"ga3","initial":"g","final":"a","tone":"3"},{"group_id":"z","group_kind":"syllable","pinyin":"a1","initial":"","final":"a","tone":"1"}]
    metric=DimsimDistance(); pairs=[]
    for i,j in ((0,1),(0,2),(1,2)):
        a,b=groups[i],groups[j]; flags=(a["initial"]==b["initial"],a["final"]==b["final"],a["tone"]==b["tone"]); mask="".join("1" if x else "0" for x in flags); pair_id=hashlib.sha256(json.dumps([a["group_id"],b["group_id"]],separators=(",",":")).encode()).hexdigest()
        pairs.append({"left_id":a["group_id"],"right_id":b["group_id"],"left_pinyin":a["pinyin"],"right_pinyin":b["pinyin"],"equal_initial":flags[0],"equal_final":flags[1],"equal_tone":flags[2],"overlap_mask":mask,"pair_id":pair_id,"left_zero_initial":False,"right_zero_initial":False,"no_zero_initial_diagnostic":True,"distance_status":"ok","dimsim_distance":metric(a["pinyin"],b["pinyin"])})
    if omit_last:pairs.pop()
    if bad_mask:pairs[0]["overlap_mask"]="000"
    (root/"pairs").mkdir(); (root/"groups.jsonl").write_text("".join(json.dumps(x)+"\n" for x in groups),encoding="utf-8"); (root/"pairs/syllable.jsonl").write_text("".join(json.dumps(x)+"\n" for x in pairs),encoding="utf-8")
    artifacts={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("groups.jsonl","pairs/syllable.jsonl")}; manifest={"schema_version":2,"evaluation_id":"synthetic","pairing_policy":"nonzero_nonzero","artifacts":artifacts}; (root/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8"); return hashlib.sha256((root/"manifest.json").read_bytes()).hexdigest()

class DimsimDistributionResearchTests(unittest.TestCase):
    def test_retained_calculations_match_original_formulas(self):
        plot,rows=retained_distribution_data(fixture_columns())
        self.assertEqual(plot["detail_values"].tolist(),[0.,5.])
        self.assertEqual(plot["full_histogram_edges"].tolist(),list(np.arange(0.,121.,10.))+[125.000001])
        self.assertEqual([rows[i]["count"] for i in (0,1,12)],[2,1,1])
        self.assertEqual(rows[0]["mean_distance"],2.5); self.assertTrue(np.isclose(rows[1]["initial_share"],5/15))
        self.assertIsNone(rows[2]["initial_share"])

    def test_component_reconstruction_and_override_guards(self):
        columns=fixture_columns(); columns["vowel_distance"][1]=3
        with self.assertRaisesRegex(ValueError,"reconstruct"): retained_distribution_data(columns)
        columns=fixture_columns(); columns["hardcode_applied"][0]=True
        with self.assertRaisesRegex(ValueError,"hardcodeMap"): retained_distribution_data(columns)

    def test_notebook_is_thin_and_excludes_removed_analysis(self):
        path=Path(__file__).parents[1]/"notebooks/DIMSIM/dimsim-distribution-analysis.ipynb"; notebook=json.loads(path.read_text(encoding="utf-8-sig")); source="".join("".join(c.get("source",[])) for c in notebook["cells"])
        self.assertLessEqual(len(notebook["cells"]),4); self.assertTrue(all(not c.get("outputs") for c in notebook["cells"] if c["cell_type"]=="code"))
        self.assertIn("run_dimsim_distribution",source); self.assertIn("EXPECTED_EVALUATION_MANIFEST_SHA256",source)
        for excluded in ("phonetic_matches","summarize_population","KDE","clustering","historical","cache","CACHE","plt.","sns."): self.assertNotIn(excluded,source)

    def test_direct_bundle_computation_has_no_hardcoded_population_or_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); digest=write_bundle(root); columns,validation=compute_retained_population(root,expected_manifest_sha256=digest)
            self.assertEqual(validation["syllable_count"],3); self.assertEqual(validation["pair_count"],3); self.assertEqual(validation["expected_pair_count"],3)
            self.assertEqual(columns["distance"].shape,(3,)); self.assertLessEqual(validation["max_formula_absolute_error"],validation["tolerance"])

    def test_direct_bundle_rejects_bad_sha_incomplete_pairs_and_mask(self):
        for kwargs,message in (({},"manifest SHA256"),({"omit_last":True},"incomplete"),({"bad_mask":True},"mask")):
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp); digest=write_bundle(root,**kwargs)
                if not kwargs:digest="0"*64
                with self.assertRaisesRegex(ValueError,message): compute_retained_population(root,expected_manifest_sha256=digest)
