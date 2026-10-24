import csv,hashlib,json,pickle,shutil,sys,tempfile,unittest,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import numpy as np
from phonetics.artifacts import sha256_file
from phonetics.preparation import prepare_acoustic_bundle,validate_acoustic_bundle,prepare_dataset,validate_prepared

class Backend:
    cache_id="fixture-v1"
    def __call__(self,entry,config):return np.array([len(entry["label"]),1.0],dtype=np.float32)

class AcousticBundleTests(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.r=Path(self.t.name);self.s=self.r/"sounds";self.s.mkdir()
  for name in ("ma1.wav","ba2.wav"):
   with wave.open(str(self.s/name),"wb") as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes((b"\x01\x00")*80)
  self.sm=self.r/"sounds.csv"
  with self.sm.open("w",encoding="utf-8",newline="") as h:w=csv.writer(h,lineterminator="\n");w.writerow(["path","pinyin"]);w.writerows([["ma1.wav","ma1"],["ba2.wav","ba2"]])
  self.model={"id":"fixture","revision":"r1","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"}
 def tearDown(self):self.t.cleanup()
 def bundle(self):
  cfg={"bundle_id":"audio-v1","output_root":str(self.r/"bundles"),"cache_root":str(self.r/"cache"),"sounds":{"path":str(self.s),"manifest":str(self.sm),"sha256":sha256_file(self.sm),"tone_convention":"explicit"},"model":self.model}
  return prepare_acoustic_bundle(cfg,embedding_backend=Backend())
 def test_bundle_portable_validated_and_pin_required(self):
  b=self.bundle();m=validate_acoustic_bundle(b);pin=sha256_file(b/"manifest.json");validate_acoustic_bundle(b,expected_manifest_sha256=pin)
  self.assertEqual(m["count"],2);self.assertEqual(m["dimension"],2);self.assertEqual(m["model"]["backend"],"stub")
  self.assertNotIn('"path":', (b/"sound_inventory.jsonl").read_text(encoding="utf-8"))
  shutil.rmtree(self.s);self.sm.unlink();validate_acoustic_bundle(b,expected_manifest_sha256=pin)
  with self.assertRaises(ValueError):validate_acoustic_bundle(b,expected_manifest_sha256="0"*64)
 def test_precomputed_dataset_needs_no_audio_or_inference(self):
  b=self.bundle();pin=sha256_file(b/"manifest.json");shutil.rmtree(self.s);self.sm.unlink()
  corpus=self.r/"corpus.csv";corpus.write_text("text,pinyin\n妈,ma1\n爸,ba2\n",encoding="utf-8")
  wubi=self.r/"wubi.pkl"
  with wubi.open("wb") as handle:pickle.dump({"妈":"abc","爸":"def"},handle)
  cfg={"dataset_id":"prepared","output_root":str(self.r/"prepared"),"corpus":{"path":str(corpus),"sha256":sha256_file(corpus),"tone_convention":"explicit"},"wubi":{"path":str(wubi),"sha256":sha256_file(wubi)},"sampling":{"size":1,"strategy":"uniform","seed":7},"acoustic_bundle":{"path":str(b),"manifest_sha256":pin}}
  out=prepare_dataset(cfg);manifest=validate_prepared(out);self.assertEqual(manifest["model"]["backend"],"precomputed");self.assertEqual(manifest["model"]["origin_backend"],"stub")
  with self.assertRaisesRegex(ValueError,"embedding_backend"):
   cfg["dataset_id"]="bad";prepare_dataset(cfg,embedding_backend=Backend())
 def test_tampered_vector_artifact_rejected(self):
  b=self.bundle();(b/"pinyin_embeddings.pkl").write_bytes(b"bad")
  with self.assertRaisesRegex(ValueError,"checksum"):validate_acoustic_bundle(b)
 def test_rehashed_semantic_backend_corruption_rejected(self):
  b=self.bundle();report=b/"reports/inference.json";inference=json.loads(report.read_text(encoding="utf-8"));inference["backend"]="transformers";report.write_text(json.dumps(inference,sort_keys=True,indent=2)+"\n",encoding="utf-8")
  manifest=json.loads((b/"manifest.json").read_text(encoding="utf-8"));manifest["inference"]=inference;manifest["artifacts"]["reports/inference.json"]=sha256_file(report);(b/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,indent=2)+"\n",encoding="utf-8")
  with self.assertRaisesRegex(ValueError,"backend"):validate_acoustic_bundle(b)

if __name__=="__main__":unittest.main()
