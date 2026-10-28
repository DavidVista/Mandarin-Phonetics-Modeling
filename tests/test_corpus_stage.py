import csv,json,shutil,sys,tempfile,unittest,wave
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phonetics.artifacts import sha256_file,write_json
from phonetics.preparation import assemble_dataset,prepare_acoustic_bundle,prepare_corpus,validate_corpus_bundle,validate_prepared
from phonetics.preparation.corpus_processing import deduplicate_rows

class Backend:
 def __init__(self,name):self.cache_id=name
 def __call__(self,entry,config):return np.array([1.,float(len(entry["label"]))],dtype=np.float32)
class CorpusStageTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.root=Path(self.tmp.name)
  self.dictionary=self.root/"dictionary.txt";self.dictionary.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in [{"character":"妈","decomposition":"？"},{"character":"爸","decomposition":"⿰父巴"},{"character":"父","decomposition":"父"},{"character":"巴","decomposition":"巴"}]),encoding="utf8")
  (self.root/"COPYING").write_text("fixture",encoding="utf8");(self.root/"LGPL").write_text("fixture",encoding="utf8")
  self.ids={"path":str(self.dictionary),"sha256":sha256_file(self.dictionary),"snapshot_commit":"618dbab8a8ddefb958763c8b4afbaa741a4460de","license":"LGPL-3.0-or-later","attribution":"fixture attribution"}
  self.sounds=self.root/"sounds";self.sounds.mkdir()
  for name in ("ma1","ma2","ba4"):
   with wave.open(str(self.sounds/f"{name}.wav"),"wb") as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\1\0"*32)
  self.sound_manifest=self.root/"sounds.csv";self.sound_manifest.write_text("path,pinyin\nma1.wav,ma1\nma2.wav,ma2\nba4.wav,ba4\n",encoding="utf8")
  self.model={"id":"fixture","revision":"r1","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"}
  self.a=self.acoustic("a","model-a")
 def tearDown(self):self.tmp.cleanup()
 def acoustic(self,bid,modelid):
  return prepare_acoustic_bundle({"bundle_id":bid,"output_root":str(self.root/"acoustic"),"cache_root":str(self.root/f"cache-{bid}"),"sounds":{"path":str(self.sounds),"manifest":str(self.sound_manifest),"sha256":sha256_file(self.sound_manifest),"tone_convention":"explicit"},"model":{**self.model,"id":modelid}},embedding_backend=Backend(bid))
 def raw(self,name,rows):
  path=self.root/name
  with path.open("w",encoding="utf8",newline="") as h:w=csv.writer(h,lineterminator="\n");w.writerow(["text","pinyin","source_id"]);w.writerows(rows)
  return path
 def corpus_config(self,path,cid="corpus",size=None,bundle=None):
  bundle=bundle or self.a
  return {"corpus_id":cid,"output_root":str(self.root/"processed"),"corpus":{"path":str(path),"sha256":sha256_file(path),"tone_convention":"explicit"},"ids":self.ids,"sampling":{"size":size,"strategy":"uniform","seed":7},"acoustic_bundle":{"path":str(bundle),"manifest_sha256":sha256_file(bundle/"manifest.json")}}
 def assembly_config(self,corpus,bundle=None,did="dataset"):
  bundle=bundle or self.a
  return {"dataset_id":did,"output_root":str(self.root/"prepared"),"processed_corpus":{"path":str(corpus),"manifest_sha256":sha256_file(corpus/"manifest.json")},"acoustic_bundle":{"path":str(bundle),"manifest_sha256":sha256_file(bundle/"manifest.json")}}
 def test_filter_dedup_sampling_and_assembly_copy(self):
  raw=self.raw("raw.csv",[["妈","ma1","1"],["妈","ma1","2"],["爸","ba4","3"],["坏","ma1","4"],["","","5"]])
  corpus=prepare_corpus(self.corpus_config(raw,size=2));manifest=validate_corpus_bundle(corpus)
  stats=json.loads((corpus/"reports/processing_statistics.json").read_text(encoding="utf8"));self.assertEqual(stats["selected"],2)
  self.assertEqual(stats["rejections"],{"ids":1,"nonempty":1})
  selected=[json.loads(x) for x in (corpus/"corpus.jsonl").read_text(encoding="utf8").splitlines()]
  self.assertEqual(next(x for x in selected if x["text"]=="妈")["source_ids"],["1","2"])
  prepared=assemble_dataset(self.assembly_config(corpus));self.assertEqual((corpus/"corpus.csv").read_bytes(),(prepared/"corpus.csv").read_bytes());validate_prepared(prepared)
  self.assertEqual(manifest["sources"]["sound_inventory_sha256"],sha256_file(prepared/"reports/sound_inventory.jsonl"))
 def test_conflicts_excluded_in_stage_and_error_in_direct_policy(self):
  raw=self.raw("conflict.csv",[["妈","ma1","1"],["妈","ma2","2"],["爸","ba4","3"],["爸","ba4","4"]])
  corpus=prepare_corpus(self.corpus_config(raw,cid="conflict"));selection=[json.loads(x) for x in (corpus/"reports/selection.jsonl").read_text(encoding="utf8").splitlines()]
  self.assertEqual([x["text"] for x in selection],["爸"]);self.assertEqual(selection[0]["source_ids"],["3","4"])
  stats=json.loads((corpus/"reports/processing_statistics.json").read_text(encoding="utf8"));stage=next(x for x in stats["stages"] if x["name"]=="conflicts")
  self.assertEqual((stage["input_count"],stage["output_count"],stage["rejected"]),(4,2,2))
  with self.assertRaisesRegex(ValueError,"conflicting labels"):deduplicate_rows([{"text":"妈","labels":["ma1"],"source_id":"1","source_row":1},{"text":"妈","labels":["ma2"],"source_id":"2","source_row":2}])
 def test_all_conflict_failure_retains_diagnostics(self):
  raw=self.raw("all-conflict.csv",[["妈","ma1","1"],["妈","ma2","2"]])
  with self.assertRaisesRegex(ValueError,"no rows remain"):prepare_corpus(self.corpus_config(raw,cid="all-conflict"))
  report=json.loads((self.root/"processed/all-conflict.failed.json").read_text(encoding="utf8"));self.assertEqual(report["details"]["conflict_rows"],2);self.assertEqual(len(report["details"]["rejections"]),2)
  self.assertFalse((self.root/"processed/all-conflict").exists())
 def test_invalid_pinyin_is_rejected_before_conflicts(self):
  raw=self.raw("invalid.csv",[["妈","ma1","1"],["妈","invalid","2"],["爸","ba4","3"]]);corpus=prepare_corpus(self.corpus_config(raw,cid="invalid"))
  stats=json.loads((corpus/"reports/processing_statistics.json").read_text(encoding="utf8"));self.assertEqual(stats["rejections"],{"pinyin":1});self.assertEqual(next(x for x in stats["stages"] if x["name"]=="conflicts")["rejected"],0)
 def test_inventory_reuse_is_model_independent_but_exact(self):
  corpus=prepare_corpus(self.corpus_config(self.raw("reuse.csv",[["妈","ma1","1"],["爸","ba4","2"]])))
  b=self.acoustic("b","model-b");validate_prepared(assemble_dataset(self.assembly_config(corpus,b,"reused")))
  other=self.root/"other";other.mkdir()
  with wave.open(str(other/"ma1.wav"),"wb") as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\2\0"*10)
  sm=self.root/"other.csv";sm.write_text("path,pinyin\nma1.wav,ma1\n",encoding="utf8")
  bad=prepare_acoustic_bundle({"bundle_id":"bad","output_root":str(self.root/"acoustic"),"cache_root":str(self.root/"cache-bad"),"sounds":{"path":str(other),"manifest":str(sm),"sha256":sha256_file(sm),"tone_convention":"explicit"},"model":self.model},embedding_backend=Backend("bad"))
  with self.assertRaisesRegex(ValueError,"sound inventory pin mismatch"):assemble_dataset(self.assembly_config(corpus,bad,"bad"))
 def test_rehashed_processing_statistics_tampering_rejected(self):
  corpus=prepare_corpus(self.corpus_config(self.raw("tamper.csv",[["妈","ma1","1"],["爸","ba4","2"]])))
  path=corpus/"reports/processing_statistics.json";stats=json.loads(path.read_text(encoding="utf8"));stats["selected"]+=1;write_json(path,stats)
  manifest_path=corpus/"manifest.json";manifest=json.loads(manifest_path.read_text(encoding="utf8"));manifest["artifacts"]["reports/processing_statistics.json"]=sha256_file(path);write_json(manifest_path,manifest)
  with self.assertRaisesRegex(ValueError,"statistics invalid"):validate_corpus_bundle(corpus)
 def test_bundle_is_portable_and_no_overwrite(self):
  corpus=prepare_corpus(self.corpus_config(self.raw("portable.csv",[["妈","ma1","1"],["爸","ba4","2"]]),cid="portable"));prepared=assemble_dataset(self.assembly_config(corpus,did="portable"))
  with self.assertRaises(FileExistsError):assemble_dataset(self.assembly_config(corpus,did="portable"))
  shutil.rmtree(self.sounds);self.sound_manifest.unlink();shutil.rmtree(self.root/"acoustic");shutil.rmtree(self.root/"processed");self.dictionary.unlink();(self.root/"COPYING").unlink();(self.root/"LGPL").unlink()
  validate_prepared(prepared)

if __name__=="__main__":unittest.main()
