from __future__ import annotations
import json,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/"src"))
from phonetics.research.artifacts import research_run

class ArtifactTests(unittest.TestCase):
 def test_complete_no_overwrite_and_failure(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/"source.py";source.write_text("x=1")
   with research_run(root,"analysis","run-1",[source],{"seed":1},[source],{"config.json","data/x.json"}) as (run,manifest):(run/"data/x.json").write_text("{}")
   final=json.loads((run/"manifest.json").read_text());self.assertEqual(final["status"],"complete");self.assertNotIn("manifest.json",[x["path"] for x in final["artifacts"]]);self.assertRaises(FileExistsError,lambda:research_run(root,"analysis","run-1",[source],{},[source],{"config.json"}).__enter__())
   with self.assertRaisesRegex(RuntimeError,"boom"):
    with research_run(root,"analysis","run-2",[source],{},[source],{"config.json"}) as (failed,_):raise RuntimeError("boom")
   self.assertEqual(json.loads((failed/"manifest.json").read_text())["status"],"failed")
 def test_output_inside_input_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/"outputs";source.mkdir();ctx=research_run(root,"x","r",[root],{},[Path(__file__)],{"config.json"})
   self.assertRaises(ValueError,ctx.__enter__)
 def test_early_failure_has_failed_manifest(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);missing=root/"input";missing.mkdir()
   with self.assertRaises(FileNotFoundError):
    with research_run(root,"analysis","early",[missing],{},[Path(__file__)],{"config.json"}):pass
   manifest=json.loads((root/"outputs/analysis/early/manifest.json").read_text());self.assertEqual(manifest["status"],"failed")
 def test_missing_or_empty_required_artifact_fails(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);source=root/"source.py";source.write_text("x")
   with self.assertRaises(RuntimeError):
    with research_run(root,"analysis","missing",[source],{},[source],{"config.json","tables/report.csv"}):pass
   self.assertEqual(json.loads((root/"outputs/analysis/missing/manifest.json").read_text())["status"],"failed")
   with self.assertRaises(RuntimeError):
    with research_run(root,"analysis","empty",[source],{},[source],{"config.json","tables/report.csv"}) as (run,_):(run/"tables/report.csv").write_text("")
   self.assertEqual(json.loads((root/"outputs/analysis/empty/manifest.json").read_text())["status"],"failed")
if __name__=="__main__":unittest.main()
