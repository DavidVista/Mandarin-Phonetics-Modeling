import csv,json,shutil,sys,tempfile,unittest,wave,types
from pathlib import Path
from unittest import mock
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import numpy as np
from phonetics.artifacts import sha256_file,write_json
from phonetics.preparation import prepare_acoustic_bundle,prepare_dataset,validate_prepared
from phonetics.preparation.config import validate_config
from phonetics.preparation.embeddings import TransformersBackend,cache_key
from phonetics.preparation.readers import audio_info

class Stub:
 cache_id="fixture-v1"
 def __call__(self,entry,config):return np.array([len(entry["label"]),1.],dtype=np.float32)
def wav(path):
 with wave.open(str(path),"wb") as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\0\0"*16)

class PreparationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.root=Path(self.temp.name)
  self.dictionary=self.root/"dictionary.txt";dictrows=[{"character":"妈","decomposition":"？"},{"character":"爸","decomposition":"⿰父巴"},{"character":"父","decomposition":"父"},{"character":"巴","decomposition":"巴"}]
  self.dictionary.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in dictrows),encoding="utf8")
  (self.root/"COPYING").write_text("fixture",encoding="utf8");(self.root/"LGPL").write_text("fixture",encoding="utf8")
  self.ids={"path":str(self.dictionary),"sha256":sha256_file(self.dictionary),"snapshot_commit":"618dbab8a8ddefb958763c8b4afbaa741a4460de","license":"LGPL-3.0-or-later","attribution":"fixture attribution"}
  self.sounds=self.root/"sounds";self.sounds.mkdir()
  for label in ("ma1","ma2","ba4"):wav(self.sounds/f"{label}.wav")
  self.sound_manifest=self.root/"sounds.csv";self.sound_manifest.write_text("path,pinyin\nma1.wav,ma1\nma2.wav,ma2\nba4.wav,ba4\n",encoding="utf8")
  self.audio=prepare_acoustic_bundle({"bundle_id":"audio","output_root":str(self.root/"audio-out"),"cache_root":str(self.root/"cache"),"sounds":{"path":str(self.sounds),"manifest":str(self.sound_manifest),"sha256":sha256_file(self.sound_manifest),"tone_convention":"explicit"},"model":{"id":"fixture","revision":"r1","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"}},embedding_backend=Stub())
  self.corpus=self.root/"corpus.csv";self.config={"dataset_id":"fixture","output_root":str(self.root/"out"),"corpus":{"path":str(self.corpus),"tone_convention":"explicit"},"ids":self.ids,"sampling":{"size":None,"strategy":"uniform","seed":7},"acoustic_bundle":{"path":str(self.audio),"manifest_sha256":sha256_file(self.audio/"manifest.json")}}
 def tearDown(self):self.temp.cleanup()
 def write_rows(self,rows):
  with self.corpus.open("w",encoding="utf8",newline="") as h:w=csv.writer(h,lineterminator="\n");w.writerow(["text","pinyin","source_id"]);w.writerows(rows)
 def make(self,did="fixture",size=None):
  return prepare_dataset({**self.config,"dataset_id":did,"sampling":{"size":size,"strategy":"uniform","seed":7}})
 def tamper_stats(self,path,mutate):
  sp=path/"reports/run_statistics.json";s=json.loads(sp.read_text(encoding="utf8"));mutate(s);write_json(sp,s)
  mp=path/"manifest.json";m=json.loads(mp.read_text(encoding="utf8"));m["artifacts"]["reports/run_statistics.json"]=sha256_file(sp);write_json(mp,m)

 def test_exact_filter_losses_and_dictionary_global_inventory(self):
  self.write_rows([["","","a"],["妈!","ma1 ma1","b"],["妈","ma1 ma1","c"],["妈","invalid","d"],["坏","ma1","e"],["妈","ma3","f"],["妈","ma1","g"],["爸","ba4","h"]])
  path=self.make();validate_prepared(path);stats=json.loads((path/"reports/processing_statistics.json").read_text(encoding="utf8"))
  rejected=stats["rejections"];self.assertEqual(rejected,{"acoustic":1,"characters":1,"ids":1,"nonempty":1,"pinyin":1,"alignment":1})
  self.assertEqual(stats["ids"]["fallback_before"]["by_mode"]["character_fallback"]["types"],1)
  self.assertEqual(stats["ids"]["fallback_after"]["by_mode"]["character_fallback"]["types"],1)
  self.assertEqual(json.loads((path/"reports/run_statistics.json").read_text(encoding="utf8"))["outputs"]["vocabulary_tokens"],5)
 def test_conflicts_fail_diagnostically_and_target_is_not_written(self):
  self.write_rows([["妈","ma1","a"],["妈","ma2","b"],["爸","ba4","c"]])
  with self.assertRaisesRegex(ValueError,"conflicting labels"):self.make()
  failure=json.loads((self.root/"out/fixture.failed.json").read_text(encoding="utf8"));self.assertEqual(failure["status"],"failure");self.assertTrue(failure["config_sha256"]);self.assertFalse((self.root/"out/fixture").exists())
 def test_sampling_overflow_overwrite_and_tampered_csv(self):
  self.write_rows([["妈","ma1","a"],["爸","ba4","b"]]);path=self.make("okay")
  with self.assertRaises(FileExistsError):self.make("okay")
  (path/"corpus.csv").write_text("broken",encoding="utf8")
  with self.assertRaisesRegex(ValueError,"artifact mismatch"):validate_prepared(path)
  with self.assertRaisesRegex(ValueError,"sampling size exceeds"):self.make("large",3)
  failure=json.loads((self.root/"out/large.failed.json").read_text(encoding="utf8"));self.assertIn("sampling",failure["message"])
 def test_rehashed_statistics_semantics_rejected(self):
  mutations=[lambda s:s["outputs"].update(rows=99),lambda s:s.pop("run_id"),lambda s:s["outputs"].update(ids_records=True),lambda s:s["outputs"].update(vocabulary_tokens=5.0)]
  for i,mutate in enumerate(mutations):
   with self.subTest(i=i):
    self.write_rows([["妈","ma1","a"],["爸","ba4","b"]]);path=self.make(f"tamper{i}");self.tamper_stats(path,mutate)
    with self.assertRaises(ValueError):validate_prepared(path)
 def test_prepared_bundle_is_portable_after_source_removal(self):
  self.write_rows([["妈","ma1","a"],["爸","ba4","b"]]);path=self.make("portable")
  shutil.rmtree(self.sounds);self.sound_manifest.unlink();shutil.rmtree(self.root/"audio-out");shutil.rmtree(self.root/"corpus-out",ignore_errors=True)
  self.dictionary.unlink();(self.root/"COPYING").unlink();(self.root/"LGPL").unlink();self.assertEqual(validate_prepared(path)["dataset_id"],"portable")
 def test_config_cache_and_truncated_wav(self):
  bad={**self.config,"sampling":{"size":True,"strategy":"uniform","seed":7}}
  with self.assertRaisesRegex(ValueError,"positive"):validate_config(bad)
  entry={"sha256":"a"*64};model={"id":"m","revision":"r","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"}
  self.assertNotEqual(cache_key(entry,model,"stub-a"),cache_key(entry,model,"transformers"))
  broken=self.root/"broken.wav";wav(broken);broken.write_bytes(broken.read_bytes()[:-2])
  with self.assertRaises((ValueError,EOFError,wave.Error)):audio_info(broken)
 def test_early_failure_reports_identity_without_success_bundle(self):
  config={**self.config,"dataset_id":"early","corpus":{**self.config["corpus"],"path":str(self.root/"missing.csv")}}
  with self.assertRaises(FileNotFoundError):prepare_dataset(config)
  report=json.loads((self.root/"out/early.failed.json").read_text(encoding="utf8"))
  self.assertEqual(report["status"],"failure");self.assertTrue(report["run_id"]);self.assertTrue(report["config_sha256"])
  self.assertFalse((self.root/"out/early").exists())
 def test_exact_full_pool_and_order_independent_sampling(self):
  rows=[["妈","ma1","a"],["爸","ba4","b"]];self.write_rows(rows)
  full=self.make("full",2);self.assertEqual(validate_prepared(full)["dataset_id"],"full")
  first=self.make("order-a",1);chosen_a=json.loads((first/"reports/selection.jsonl").read_text(encoding="utf8"))
  self.write_rows(list(reversed(rows)));second=self.make("order-b",1);chosen_b=json.loads((second/"reports/selection.jsonl").read_text(encoding="utf8"))
  self.assertEqual(chosen_a["sample_id"],chosen_b["sample_id"])
 def test_real_backend_padding_layer_pool_and_resample(self):
  try: import torch
  except ImportError: self.skipTest("torch is unavailable in this runtime")
  calls={}
  class Processor:
   def __call__(self,samples,**kwargs):calls["samples"]=np.asarray(samples);calls.update(kwargs);return {"input_values":torch.tensor([samples],dtype=torch.float32)}
  class Model:
   def __call__(self,**kwargs):calls["inference_mode"]=torch.is_inference_mode_enabled();return types.SimpleNamespace(hidden_states=(torch.zeros(1,2,2),torch.tensor([[[1.,3.],[5.,7.]]])))
  backend=TransformersBackend.__new__(TransformersBackend);backend.torch=torch;backend.processor=Processor();backend.model=Model();config={"sample_rate":4,"padding":"legacy_one_second","layer":1}
  with mock.patch.dict(TransformersBackend.__call__.__globals__,{"_read_audio":mock.Mock(return_value=(np.array([.1,.2],dtype=np.float32),4))}):vector=backend({"path":"x.wav"},config)
  self.assertEqual(len(calls["samples"]),4);self.assertTrue(calls["inference_mode"]);np.testing.assert_allclose(vector,[3.,5.])
  fake=types.SimpleNamespace(resample=mock.Mock(return_value=np.array([.4,.5],dtype=np.float32)))
  with mock.patch.dict(TransformersBackend.__call__.__globals__,{"_read_audio":mock.Mock(return_value=(np.array([.1],dtype=np.float32),2))}),mock.patch.dict(sys.modules,{"librosa":fake}):backend({"path":"x.wav"},{**config,"padding":"none"})
  fake.resample.assert_called_once_with
  self.assertEqual(fake.resample.call_args.kwargs,{"orig_sr":2,"target_sr":4})

if __name__=="__main__":unittest.main()
