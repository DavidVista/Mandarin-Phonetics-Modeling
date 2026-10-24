import ast,csv,hashlib,importlib.util,json,pickle,shutil,sys,tempfile,unittest,wave
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parents[1]/"src"))
import numpy as np
from phonetics.artifacts import sha256_file
from phonetics.preparation import prepare_dataset,validate_acoustic_bundle,validate_prepared
ROOT=Path(__file__).parents[2];BUILDER=ROOT/"meta/tools/build_preparation_notebook.py";FOLDER=ROOT/"project/notebooks/Datasets/Data Preparation";NOTEBOOK=FOLDER/"prepare-pinyin-embeddings.ipynb";COMPOSITE=FOLDER/"prepare-pinyin-embeddings-composite.ipynb"
class Stub:
 cache_id="portable-fixture"
 def __call__(self,entry,config):return np.array([1.0,len(entry["label"])],dtype=np.float32)
class PortableNotebookTests(unittest.TestCase):
 def builder(self):
  spec=importlib.util.spec_from_file_location("portable_builder",BUILDER);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
 def execute_definitions(self):
  nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"));ns={}
  for cell in nb["cells"]:
   if cell["cell_type"]=="code" and cell.get("id")!="inference":exec("".join(cell["source"]),ns)
  return nb,ns
 def test_readable_sections_and_builder_outputs(self):
  m=self.builder();nb=json.loads(NOTEBOOK.read_text(encoding="utf-8"));text=NOTEBOOK.read_text(encoding="utf-8")
  self.assertGreaterEqual(sum(c["cell_type"]=="markdown" for c in nb["cells"]),10);self.assertGreaterEqual(sum(c["cell_type"]=="code" for c in nb["cells"]),12)
  for forbidden in ("EMBEDDED_SOURCES","sys.path","importlib.import_module","from phonetics","PROJECT_ROOT","__file__"):self.assertNotIn(forbidden,text)
  for cell in nb["cells"]:
   if cell["cell_type"]=="code":
    tree=ast.parse("".join(cell["source"]));self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in {"exec","eval"} for node in ast.walk(tree)))
  _,ns=self.execute_definitions()
  if "RUN_INFERENCE" in ns:self.assertFalse(ns["RUN_INFERENCE"])
  expected={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in m.SDK_FILES.items()};self.assertEqual(ns["BUILD_METADATA"]["original_sdk_source_sha256"],dict(sorted(expected.items())))
  code_by_id={c["id"]:"".join(c["source"]) for c in nb["cells"] if c["cell_type"]=="code"}
  self.assertEqual(ns["BUILD_METADATA"]["inline_source_sha256"],{key:hashlib.sha256(code_by_id[key].encode()).hexdigest() for key in ns["BUILD_METADATA"]["inline_source_sha256"]})
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
   td=Path(td);m.DATA_TARGET=td/"data.ipynb";m.SOUNDS_TARGET=td/"sounds.ipynb";m.CORPUS_TARGET=td/"corpus.ipynb";m.COMPOSITE_TARGET=td/"composite.ipynb";m.PORTABLE_TARGET=td/"portable.ipynb";m.build()
   for actual,generated in ((FOLDER/"prepare-data.ipynb",m.DATA_TARGET),(FOLDER/"prepare-sounds.ipynb",m.SOUNDS_TARGET),(FOLDER/"prepare-corpus.ipynb",m.CORPUS_TARGET),(COMPOSITE,m.COMPOSITE_TARGET),(NOTEBOOK,m.PORTABLE_TARGET)):self.assertEqual(actual.read_bytes(),generated.read_bytes())
 def test_inline_bundle_is_consumed_offline(self):
  _,ns=self.execute_definitions()
  with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as td:
   root=Path(td);sounds=root/"sounds";sounds.mkdir()
   for name in ("ma1.wav","ba2.wav"):
    with wave.open(str(sounds/name),"wb") as h:h.setnchannels(1);h.setsampwidth(2);h.setframerate(8000);h.writeframes(b"\x01\x00"*80)
   sm=root/"sounds.csv"
   with sm.open("w",encoding="utf-8",newline="") as h:w=csv.writer(h,lineterminator="\n");w.writerow(["path","pinyin"]);w.writerows([["ma1.wav","ma1"],["ba2.wav","ba2"]])
   model={"id":"fixture","revision":"r1","layer":-1,"pooling":"mean_all","sample_rate":8000,"mono":True,"padding":"none","dtype":"float32","device":"cpu"};cfg={"bundle_id":"portable","output_root":str(root/"bundles"),"cache_root":str(root/"cache"),"sounds":{"path":str(sounds),"manifest":str(sm),"sha256":sha256_file(sm),"tone_convention":"explicit"},"model":model}
   bundle=ns["prepare_acoustic_bundle"](cfg,embedding_backend=Stub());pin=sha256_file(bundle/"manifest.json");validate_acoustic_bundle(bundle,expected_manifest_sha256=pin);shutil.rmtree(sounds);sm.unlink()
   corpus=root/"corpus.csv";corpus.write_text("text,pinyin\n妈,ma1\n爸,ba2\n",encoding="utf-8");wubi=root/"wubi.pkl"
   with wubi.open("wb") as h:pickle.dump({"妈":"abc","爸":"def"},h)
   prepared=prepare_dataset({"dataset_id":"ready","output_root":str(root/"prepared"),"corpus":{"path":str(corpus),"sha256":sha256_file(corpus),"tone_convention":"explicit"},"wubi":{"path":str(wubi),"sha256":sha256_file(wubi)},"sampling":{"size":None,"strategy":"uniform","seed":1},"acoustic_bundle":{"path":str(bundle),"manifest_sha256":pin}});validate_prepared(prepared)
if __name__=="__main__":unittest.main()
