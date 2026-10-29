import csv,json,os,shutil,sys,tempfile,unittest,wave
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
from phonetics.artifacts import sha256_file,write_json
from phonetics.ids import build_mapping,parse_expression,read_dictionary
from phonetics.preparation import (assemble_dataset,prepare_acoustic_bundle,prepare_corpus,prepare_dataset,validate_corpus_bundle,validate_prepared)

class Backend:
 cache_id="ids-fixture"
 def __call__(self,entry,config):return np.array([1.,float(len(entry["label"]))],dtype=np.float32)

class IDSPreparationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.root=Path(self.temp.name)
  self.dictionary=self.root/"dictionary.txt"
  rows=[{"character":"妈","decomposition":"⿰女？"},{"character":"爸","decomposition":"⿱父巴"},{"character":"女","decomposition":"女"},{"character":"父","decomposition":"父"},{"character":"巴","decomposition":"巴"},{"character":"丁","decomposition":"一"},{"character":"一","decomposition":"一"},{"character":"火","decomposition":"火"},{"character":"木","decomposition":"木"}]
  self.dictionary.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
  (self.root/"COPYING").write_text("fixture COPYING",encoding="utf8");(self.root/"LGPL").write_text("fixture LGPL",encoding="utf8")
  self.ids={"path":str(self.dictionary),"sha256":sha256_file(self.dictionary),"snapshot_commit":"618dbab8a8ddefb958763c8b4afbaa741a4460de","license":"LGPL-3.0-or-later","attribution":"fixture source attribution"}
  sounds=self.root/"sounds";sounds.mkdir()
  for label in ("ma1","ma3","ba4"):
   with wave.open(str(sounds/f"{label}.wav"),"wb") as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\x01\x00"*32)
  self.sound_root=sounds;self.sound_manifest=self.root/"sounds.csv"
  self.sound_manifest.write_text("path,pinyin\nma1.wav,ma1\nma3.wav,ma3\nba4.wav,ba4\n",encoding="utf8")
  self.model={"id":"fixture","revision":"r1","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"}
  self.acoustic=prepare_acoustic_bundle({"bundle_id":"audio","output_root":str(self.root/"audio-bundles"),"cache_root":str(self.root/"cache"),"sounds":{"path":str(sounds),"manifest":str(self.sound_manifest),"sha256":sha256_file(self.sound_manifest),"tone_convention":"explicit"},"model":self.model},embedding_backend=Backend())
 def tearDown(self):self.temp.cleanup()
 def corpus(self,name,rows):
  path=self.root/name
  with path.open("w",encoding="utf8",newline="") as h:
   w=csv.writer(h,lineterminator="\n");w.writerow(["text","pinyin","source_id"]);w.writerows(rows)
  return path
 def cfg(self,path,cid="processed",size=None):
  return {"corpus_id":cid,"output_root":str(self.root/"corpus-out"),"corpus":{"path":str(path),"sha256":sha256_file(path),"tone_convention":"explicit"},"ids":self.ids,"sampling":{"size":size,"strategy":"uniform","seed":5},"acoustic_bundle":{"path":str(self.acoustic),"manifest_sha256":sha256_file(self.acoustic/"manifest.json")}}
 def test_prefix_grammar_and_fallback(self):
  self.assertEqual(parse_expression("⿰木⿱日月"),tuple("⿰木⿱日月"))
  self.assertEqual(parse_expression("⿲木火水"),tuple("⿲木火水"))
  for expr in ("", "⿰木", "⿰木火水", "⿼木火", "㇯", "⿰㇯木"):
   with self.subTest(expr=expr),self.assertRaises(ValueError):parse_expression(expr)
  duplicate=self.root/"duplicate.jsonl";duplicate.write_text('{"character":"木","decomposition":"木"}\n{"character":"木","decomposition":"火"}\n',encoding="utf8")
  with self.assertRaisesRegex(ValueError,"duplicate"):read_dictionary(duplicate)
  empty=self.root/"empty.jsonl";empty.write_text('{"character":"木","decomposition":""}\n',encoding="utf8")
  with self.assertRaisesRegex(ValueError,"empty"):read_dictionary(empty)
  mapping,vocab,_=build_mapping(self.ids)
  self.assertEqual(mapping["characters"]["妈"]["tokens"],["妈"])
  self.assertEqual(mapping["characters"]["爸"]["tokens"],["⿱","父","巴"])
  self.assertEqual(mapping["characters"]["丁"]["tokens"],["一"])
  self.assertEqual(vocab["tokens"][0],"[PAD]");self.assertEqual(vocab["stoi"]["[PAD]"],0)
  self.assertEqual(sorted(vocab["tokens"][1:],key=ord),vocab["tokens"][1:])
 def test_bundled_mapping_preserves_foreign_source_paths_and_checks_hash(self):
  expected_mapping,expected_vocab,_=build_mapping(self.ids)
  resolve=Path.resolve
  for source in (r"Z:\unavailable-client\dictionary.txt","/unavailable-client/dictionary.txt"):
   with self.subTest(source=source):
    pin={**self.ids,"path":source}
    def resolve_bundle_only(path,*args,**kwargs):
     if str(path)==source:raise AssertionError("Historical source path must not be resolved")
     return resolve(path,*args,**kwargs)
    with patch.object(Path,"resolve",resolve_bundle_only):
     mapping,vocab,provenance=build_mapping(pin,dictionary_path=self.dictionary)
    self.assertEqual(provenance["path"],source)
    self.assertEqual(mapping,{**expected_mapping,"provenance":{**expected_mapping["provenance"],"path":source}})
    self.assertEqual(vocab,expected_vocab)
    with self.assertRaisesRegex(ValueError,"checksum mismatch"):
     build_mapping({**pin,"sha256":"0"*64},dictionary_path=self.dictionary)
 def test_relative_and_forward_slash_ids_paths_survive_bundle_relocation(self):
  raw=self.corpus("portable-paths.csv",[["妈","ma1","a"],["爸","ba4","b"]])
  for index,source in enumerate((os.path.relpath(self.dictionary,Path.cwd()),self.dictionary.resolve().as_posix())):
   with self.subTest(source=source):
    cfg=self.cfg(raw,f"paths-{index}");cfg["ids"]={**self.ids,"path":source}
    corpus=prepare_corpus(cfg)
    prepared=assemble_dataset({"dataset_id":f"prepared-{index}","output_root":str(self.root/"prepared"),"processed_corpus":{"path":str(corpus),"manifest_sha256":sha256_file(corpus/"manifest.json")},"acoustic_bundle":cfg["acoustic_bundle"]})
    moved=self.root/f"moved-{index}";shutil.move(str(corpus),moved)
    resolve=Path.resolve
    original_path=str(self.dictionary.resolve())
    def resolve_without_source(path,*args,**kwargs):
     if str(path) in {source,original_path}:raise AssertionError("Historical source path must not be resolved")
     return resolve(path,*args,**kwargs)
    with patch.object(Path,"resolve",resolve_without_source):
     validate_corpus_bundle(moved)
     validate_prepared(prepared)
 def test_missing_rows_coverage_dedup_and_staged_prepared(self):
  raw=self.corpus("input.csv",[["妈","ma1","a"],["爸","ba4","b"],["妈爸","ma1 ba4","c"],["坏","ma1","d"],["爸","ba4","e"]])
  corpus=prepare_corpus(self.cfg(raw));cm=validate_corpus_bundle(corpus)
  selection=[json.loads(x) for x in (corpus/"reports/selection.jsonl").read_text(encoding="utf8").splitlines()]
  self.assertEqual(len(selection),3);both=next(x for x in selection if x["text"]=="爸")
  self.assertEqual(both["source_ids"],["b","e"]);self.assertEqual(both["sentence_length"],1);self.assertEqual(both["positions"],[0])
  self.assertEqual(len(both["ids_sequences"]),1);self.assertTrue(all(i>0 for s in both["ids_sequences"] for i in s))
  rejections=[json.loads(x) for x in (corpus/"reports/rejections.jsonl").read_text(encoding="utf8").splitlines()]
  miss=next(r for r in rejections if r["reason"]=="ids");self.assertEqual(miss["source_id"],"d");self.assertEqual(miss["missing_characters"],["坏"])
  prepared=assemble_dataset({"dataset_id":"prepared","output_root":str(self.root/"prepared"),"processed_corpus":{"path":str(corpus),"manifest_sha256":sha256_file(corpus/"manifest.json")},"acoustic_bundle":self.cfg(raw)["acoustic_bundle"]})
  self.assertTrue(validate_prepared(prepared)["ids_provenance"]);self.assertEqual(cm["schema_version"],2)
  self.assertFalse((prepared/"corpus/train.csv").exists());self.assertFalse((prepared/"wubi").exists())
 def test_vocab_is_dictionary_global_and_direct_matches_staged(self):
  raw=self.corpus("order.csv",[["妈","ma1","1"],["爸","ba4","2"]]);cfg=self.cfg(raw,"stage")
  staged=prepare_corpus(cfg);direct=prepare_dataset({"dataset_id":"direct","output_root":str(self.root/"direct"),"corpus":cfg["corpus"],"ids":self.ids,"sampling":cfg["sampling"],"acoustic_bundle":cfg["acoustic_bundle"]})
  self.assertEqual((staged/"ids/vocabulary.json").read_bytes(),(direct/"ids/vocabulary.json").read_bytes())
  self.assertEqual((staged/"corpus.jsonl").read_bytes(),(direct/"corpus.jsonl").read_bytes())
  saved_config=json.loads((direct/"reports/processed_corpus_config.json").read_text(encoding="utf8"))
  self.assertEqual(saved_config["sampling"],cfg["sampling"]);self.assertEqual(saved_config["conflict_policy"],"error")
  self.assertEqual(json.loads((staged/"config.json").read_text(encoding="utf8"))["conflict_policy"],"exclude")
  other=self.corpus("reverse.csv",[["爸","ba4","2"],["妈","ma1","1"]]);staged2=prepare_corpus(self.cfg(other,"reverse"))
  self.assertEqual((staged/"ids/vocabulary.json").read_bytes(),(staged2/"ids/vocabulary.json").read_bytes())
  with self.assertRaises(FileExistsError):prepare_corpus(cfg)
 def test_invalid_dictionary_pin_and_tampered_sequences_fail(self):
  broken=self.root/"broken.jsonl";broken.write_text(json.dumps({"character":"坏","decomposition":"⿰？"},ensure_ascii=False)+"\n",encoding="utf8")
  bad={**self.ids,"path":str(broken),"sha256":sha256_file(broken)}
  with self.assertRaisesRegex(ValueError,"too few operands"):build_mapping(bad)
  raw=self.corpus("tamper.csv",[["妈","ma1","1"],["爸","ba4","2"]]);corpus=prepare_corpus(self.cfg(raw,"tamper"))
  seqpath=corpus/"corpus.jsonl";rows=[json.loads(x) for x in seqpath.read_text(encoding="utf8").splitlines()];rows[0]["ids_sequences"][0][0]=True
  tampered="".join(json.dumps(x,ensure_ascii=False,sort_keys=True)+"\n" for x in rows);seqpath.write_text(tampered,encoding="utf8");(corpus/"reports/selection.jsonl").write_text(tampered,encoding="utf8")
  manifest=json.loads((corpus/"manifest.json").read_text(encoding="utf8"));manifest["artifacts"]["corpus.jsonl"]=sha256_file(seqpath);manifest["artifacts"]["reports/selection.jsonl"]=sha256_file(corpus/"reports/selection.jsonl");write_json(corpus/"manifest.json",manifest)
  with self.assertRaises(ValueError):validate_corpus_bundle(corpus)

if __name__=="__main__":unittest.main()
